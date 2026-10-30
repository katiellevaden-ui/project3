#!/usr/bin/env python3
"""Measure control system prompts N times for one model, so the constitution effect can be
separated from the effect of having any system prompt.

    python scripts/controls.py --model claude-sonnet-5 --judge openrouter/deepseek/deepseek-v4-flash-0731 \\
        --controls constitutions/controls/placebo_format.md constitutions/c0_general_assistant.md --repeats 10 --out runs/controls
Logs: <out>/logs/<model>/<control>/repeat<k>; the analysis is in scripts/controls_report.py.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run  # noqa: E402
from valuedrift.registry import ROOT, load_models, safe_name  # noqa: E402
from valuedrift.valuebench import valuebench_task  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True); ap.add_argument("--judge", required=True)
ap.add_argument("--controls", nargs="+", required=True); ap.add_argument("--repeats", type=int, default=10)
ap.add_argument("--questionnaires", default="PVQ40,MFT23,LVI,HEXACO"); ap.add_argument("--split", default="measure")
ap.add_argument("--vb-epochs", type=int, default=1); ap.add_argument("--reasoning", default=None)
ap.add_argument("--timeout", type=int, default=600); ap.add_argument("--max-connections", type=int, default=10)
ap.add_argument("--display", default="plain"); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--models-file", default=str(ROOT / "models.yaml")); ap.add_argument("--out", default=str(ROOT / "runs" / "controls"))
args = ap.parse_args()
m = load_models(Path(args.models_file), nicks=[args.model])[0]
reasoning = args.reasoning or m.reasoning_or("none")
for c in args.controls:
    name = Path(c).stem
    for k in range(args.repeats):
        print(f"[control] {m.nick} / {name} (repeat {k})")
        label = json.dumps({"model": m.nick, "constitution": f"control:{name}", "chain": None, "round": None, "repeat": k})
        run(valuebench_task(constitution=c, questionnaires=args.questionnaires, split=args.split, label=label, reasoning=reasoning),
            Path(args.out) / "logs" / safe_name(m.nick) / safe_name(name) / f"repeat{k}", m.inspect_name, {"judge": args.judge}, args, m.model_args,
            epochs=args.vb_epochs)
