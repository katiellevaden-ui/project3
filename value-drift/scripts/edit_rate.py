"""How often does a checkpoint decline to edit? Re-run the bench loop's edit stage K times on a
finished run's checkpoint with the same constitution and the same evidence block, and tabulate
the decision kinds (edit / unchanged / failure).

    /workspace/venv/bin/python scripts/edit_rate.py --run runs/bench-oct-001 --round 2 --samples 10

Serves the checkpoint once (the round's current checkpoint from state.json, i.e. the model that
made the decision), then calls the edit stage K times at the config's sampling settings. Each
sample's review is written under <run>/edit_rate/round_<n>/sample_<k>/review.json and a summary
to <run>/edit_rate/round_<n>/summary.json. Diagnostic only: nothing in the run's state changes.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recursive_oct.bench_loop import edit  # noqa: E402
from recursive_oct.serving import ServedModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--round", type=int, required=True, help="the round whose edit stage to replay (its evidence.txt must exist)")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--checkpoint", default=None, help="override the checkpoint (default: state.json current_checkpoint)")
    a = ap.parse_args()
    root = a.run
    cfg = json.loads((root / "config.json").read_text())
    state = json.loads((root / "state.json").read_text())
    bench = dict(cfg["bench"])
    if "invitation_path" in bench:
        bench["invitation_text"] = Path(bench["invitation_path"]).read_text()
    rd = root / f"round_{a.round:03d}"
    evidence = (rd / "evidence.txt").read_text()
    constitution = Path(state["current_constitution"])
    checkpoint = a.checkpoint or state["current_checkpoint"]
    out = root / "edit_rate" / f"round_{a.round:03d}"
    out.mkdir(parents=True, exist_ok=True)
    kinds = collections.Counter()
    rows = []
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    with ServedModel(checkpoint, cfg["serve"], f"M{a.round - 1:03d}", log_dir=out / "serve-logs") as served:
        for k in range(a.samples):
            sd = out / f"sample_{k:02d}"
            if (sd / "review.json").exists():
                review = json.loads((sd / "review.json").read_text())
            else:
                started = time.time()
                review = edit(served, constitution, evidence, sd / "edit", bench)
                review["seconds"] = time.time() - started
                sd.mkdir(parents=True, exist_ok=True)
                (sd / "review.json").write_text(json.dumps(review, indent=1))
            kinds[review["kind"]] += 1
            rows.append({"sample": k, "kind": review["kind"], "status": review["status"],
                         "words": len((review.get("text") or "").split()),
                         "justification": (review.get("justification") or "")[:400]})
            print(f"[edit_rate] sample {k}: {review['kind']}", flush=True)
    summary = {"run": str(root), "round": a.round, "checkpoint": checkpoint, "constitution": str(constitution),
               "samples": a.samples, "kinds": dict(kinds),
               "unchanged_rate": kinds.get("unchanged", 0) / a.samples, "edit_rate": kinds.get("edit", 0) / a.samples,
               "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
