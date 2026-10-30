#!/usr/bin/env python3
"""Recursive self-editing of a constitution: run chains, export every version, report each round.

    python scripts/recursive_chain.py --model claude-sonnet-5 \\
        --constitution runs/elicit-study/elicitations/claude-sonnet-5/actual-none-plain/r0.md \\
        --rounds 3 --chains 1 --out runs/chain-smoke

Each round the model is shown the current constitution as its own and may return it unchanged
or revised (valuedrift.chain, objective_editing's reflective prompt). Versions go to
<out>/versions/<model>/<constitution>/chain<k>/round<t>.md; the report lists, per round, the
model's decision, how much of the text changed, and the justification it gave.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval

from objective_editing.metrics import change_ratio
from valuedrift.chain import chain_task, export_versions
from valuedrift.registry import ROOT, load_models, safe_name

load_dotenv(ROOT / ".env")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="nick from models.yaml")
    ap.add_argument("--constitution", required=True, help="path to the starting constitution (.md)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--chains", type=int, default=1, help="independent chains from the same start")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--reasoning", default=None, help="override the model's / default reasoning setting")
    ap.add_argument("--out", default=str(ROOT / "runs" / "chain-smoke"))
    ap.add_argument("--models-file", default=str(ROOT / "models.yaml"))
    ap.add_argument("--display", default="plain")
    args = ap.parse_args()

    m = load_models(Path(args.models_file), nicks=[args.model])[0]
    reasoning = args.reasoning or m.reasoning_or("none")
    c_path = Path(args.constitution)
    c_name = "-".join(c_path.parts[-3:]).removesuffix(".md") if "elicitations" in c_path.parts else c_path.stem
    out = Path(args.out)
    label = json.dumps({"model": m.nick, "constitution": c_name, "chain": None, "round": None, "repeat": 0, "source": str(c_path)})
    print(f"[chain] {m.nick} on {c_path} | {args.rounds} rounds x {args.chains} chain(s) | reasoning={reasoning}")
    [log] = inspect_eval(chain_task(constitution=str(c_path), rounds=args.rounds, max_tokens=args.max_tokens, label=label, reasoning=reasoning),
                         model=m.inspect_name, model_args=m.model_args, log_dir=str(out / "logs" / safe_name(m.nick) / safe_name(c_name)),
                         epochs=args.chains, max_retries=3, timeout=600, display=args.display)
    if log.status != "success":
        raise SystemExit(f"chain failed: {log.error}")
    index = export_versions(log, out / "versions", m.nick, c_name)

    for sample in log.samples:
        st = sample.store
        versions, kinds, justs = st.get("ChainStore:versions", []), st.get("ChainStore:kinds", []), st.get("ChainStore:justifications", [])
        print(f"\n=== chain {sample.epoch}: {len(kinds)} rounds, {sum(k == 'edit' for k in kinds)} edits, "
              f"final text {change_ratio(versions[0], versions[-1]):.0%} changed from the start ===")
        for t, (kind, just) in enumerate(zip(kinds, justs), start=1):
            prev, cur = versions[t - 1], versions[t]
            print(f"round {t}: {kind:9} | change from previous {change_ratio(prev, cur):5.0%} | words {len(prev.split())} -> {len(cur.split())}")
            print(f"         {just.strip()[:300]}")
        if st.get("ChainStore:error"):
            print("   ERROR:", st["ChainStore:error"])
    print(f"\nversions written: {len(index)} files under {out / 'versions' / safe_name(m.nick) / safe_name(c_name)}")


if __name__ == "__main__":
    main()
