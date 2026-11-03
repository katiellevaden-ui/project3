#!/usr/bin/env python3
"""Recursive constitution editing with OCT training, measured as in the inference study.

    python scripts/run_oct_bench.py --config configs/bench-oct-001.json --run runs/bench-oct-001
    python scripts/run_oct_bench.py --config ... --run ... --resume
    python scripts/run_oct_bench.py --config ... --run ... --stages report

Requires the inference bench (value-drift-bench and objective-editing) installed in the same
environment (scripts/bootstrap_bench.sh) and OPENROUTER_API_KEY for the judge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recursive_oct.bench_loop import FakeTrainer, OCTTrainer, bench_report, run_bench_trajectory  # noqa: E402
from recursive_oct.budget import can_afford  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True); p.add_argument("--run", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stages", nargs="*", default=["loop", "report"])
    p.add_argument("--ledger", default=None, help="spending ledger for the budget guard (optional)")
    p.add_argument("--no-eigen", action="store_true")
    p.add_argument("--trainer", default="oct", choices=["oct", "fake"], help="fake: no training, checkpoints are copied (offline plumbing check)")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if not cfg.get("frozen"):
        raise ValueError("Freeze the config (\"frozen\": true) before a run; smoke configs may set it too")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    def budget_ok():
        if not args.ledger:
            return True
        return can_afford(json.loads(Path(args.ledger).read_text()), time.time(), cfg.get("stage_cost_margin_usd", 2))
    result = None
    if "loop" in args.stages:
        trainer = OCTTrainer(cfg) if args.trainer == "oct" else FakeTrainer()
        result = run_bench_trajectory(args.run, cfg, trainer, resume=args.resume, budget_ok=budget_ok)
        print(json.dumps({k: v for k, v in result.items() if k != "reviews"}, indent=1), flush=True)
    failed = bool(result and result["status"] in {"EDITING_FAILURE", "TRAINING_FAILURE"})
    if "report" in args.stages and failed:
        print("loop failed; skipping the report stage", flush=True)
    elif "report" in args.stages:
        steps = bench_report(args.run, cfg, eigen=not args.no_eigen)
        print(steps.round(3).to_string(index=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
