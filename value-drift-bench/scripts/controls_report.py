#!/usr/bin/env python3
"""Separate the constitution's effect from the effect of any system prompt, for one model.

    python scripts/controls_report.py --model claude-sonnet-5 --controls-logs runs/controls/logs \\
        --study-logs runs/elicit-study/logs/valuebench --out runs/controls/results

States (answers pooled per item): the prior (study, 10 runs); each control (N runs); the model's
own elicited constitutions (study, 30 pooled). Every pair is compared with the permutation W1
test, so each distance comes with its no-effect floor, excess and p-value."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log
from valuedrift import elicit_analysis as ea
from valuedrift.closed_loop import prior_answers_from_study, vb_answers

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True); ap.add_argument("--controls-logs", required=True); ap.add_argument("--study-logs", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n-perm", type=int, default=300)
args = ap.parse_args()

states: dict[str, dict[str, np.ndarray]] = {}
prior, _ = prior_answers_from_study(Path(args.study_logs), args.model); states["prior"] = prior
df = ea.load_study(Path(args.study_logs)); df = df[(df.model == args.model) & (df.constitution != "none") & (df["repeat"].fillna(0) == 0)]
states["own constitutions (30 pooled)"] = {i: g["oriented"].to_numpy() for i, g in df.groupby("item")}
for d in sorted(Path(args.controls_logs, args.model).iterdir()):
    pooled: dict[str, list[float]] = {}
    for info in list_eval_logs(str(d), recursive=True):
        log = read_eval_log(info.name)
        if log.status != "success": continue
        for i, v in vb_answers(log)[0].items(): pooled.setdefault(i, []).extend(v)
    states[f"control: {d.name}"] = {i: np.array(v) for i, v in pooled.items()}
rows = []
names = list(states)
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        t = ea.permutation_w1(states[names[a]], states[names[b]], n_perm=args.n_perm)
        rows.append({"state_a": names[a], "state_b": names[b], "n_a": int(np.median([len(v) for v in states[names[a]].values()])),
                     "n_b": int(np.median([len(v) for v in states[names[b]].values()])), **t})
out = pd.DataFrame(rows); Path(args.out).mkdir(parents=True, exist_ok=True); out.to_csv(Path(args.out) / "controls.csv", index=False)
pd.set_option("display.width", 200)
print(out[["state_a", "state_b", "n_a", "n_b", "observed_w1", "null_mean", "excess", "p_value"]].round(3).to_string(index=False))
