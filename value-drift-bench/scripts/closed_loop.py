#!/usr/bin/env python3
"""Closed-loop recursive self-editing with ValueBench feedback and EigenBench scoring.

    python scripts/closed_loop.py --model claude-sonnet-5 \\
        --constitution runs/elicit-study/elicitations/claude-sonnet-5/actual-none-plain/r0.md \\
        --prompts runs/edit-prompts/edit_prompts.json --invitations reflective gen07 gen10 \\
        --judge openrouter/deepseek/deepseek-v4-flash-0731 --rounds 3 --chains 3 --out runs/closed-loop

Per (invitation, chain), for t = 0..rounds-1:
    measure   ValueBench under version t (the current text)                 -> logs/<inv>/chain<k>/round<t>/vb
    edit      show the model that measurement against its prior, invite it  -> logs/<inv>/chain<k>/round<t>/edit
              to edit under the invitation; the reply is version t+1
and ValueBench under the final version, so every version is measured. Then EigenBench: every
version's responses to the scenarios, one judge batch against --reference (default: the
starting constitution, so the score reads as alignment to the model's original self-written
values), EigenTrust scores for every version plus the prior and the start.

Every step resumes from its log directory. Outputs: versions/<inv>/chain<k>/round<t>.md,
results/steps.csv, results/eigen_scores.json, results/figures/loop_editing.png and
results/figures/loop_scores.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.log import list_eval_logs, read_eval_log

from objective_editing.data import parse_principles
from objective_editing.metrics import change_ratio
from valuedrift import closed_loop as cl
from valuedrift import elicit_analysis as ea
from valuedrift.chain import chain_task
from valuedrift.eigen import (build_evaluations, eigen_judge_task, eigen_responses_task, fit_scores, load_scenarios, make_pairs,
                              responses_from_log)
from valuedrift.registry import ROOT, load_models, safe_name
from valuedrift.valuebench import valuebench_task

load_dotenv(ROOT / ".env")


def done(log_dir: Path):
    if not log_dir.exists():
        return None
    for info in list_eval_logs(str(log_dir)):
        if read_eval_log(info.name, header_only=True).status == "success":
            return read_eval_log(info.name)
    return None


def run(task, log_dir: Path, model, roles, args, model_args=None, **kw):
    existing = done(log_dir)
    if existing is not None:
        return existing
    [log] = inspect_eval(task, model=model, model_args=model_args or {}, model_roles=roles or {}, log_dir=str(log_dir),
                         timeout=args.timeout, max_retries=3, max_connections=args.max_connections, display=args.display, **kw)
    if log.status != "success":
        raise SystemExit(f"FAILED {log_dir}: {log.error}")
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--constitution", required=True, help="starting constitution (.md)")
    ap.add_argument("--prompts", required=True, help="edit_prompts.json from scripts/edit_prompt_study.py")
    ap.add_argument("--invitations", nargs="*", default=["reflective", "gen07", "gen10"])
    ap.add_argument("--judge", required=True, help="ValueBench evaluator and EigenBench judge")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--chain-start", type=int, default=1, help="number chains from here (run chains in separate processes)")
    ap.add_argument("--questionnaires", default="PVQ40,MFT23,LVI,HEXACO")
    ap.add_argument("--split", default="measure")
    ap.add_argument("--vb-epochs", type=int, default=1)
    ap.add_argument("--study-logs", default=str(ROOT / "runs" / "elicit-study" / "logs" / "valuebench"), help="where the prior runs are")
    ap.add_argument("--reference", default=None, help="EigenBench judging constitution (default: the starting constitution)")
    ap.add_argument("--scenarios-limit", type=int, default=20)
    ap.add_argument("--random-pairs", type=int, default=2)
    ap.add_argument("--stages", nargs="*", default=["loop", "eigen", "report"])
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-connections", type=int, default=10)
    ap.add_argument("--display", default="plain")
    ap.add_argument("--models-file", default=str(ROOT / "models.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "closed-loop"))
    args = ap.parse_args()

    out = Path(args.out)
    m = load_models(Path(args.models_file), nicks=[args.model])[0]
    reasoning = args.reasoning or m.reasoning_or("none")
    roles = {"judge": args.judge}
    prompts = json.load(open(args.prompts))["prompts"]
    start_text = Path(args.constitution).read_text(encoding="utf-8").strip()
    prior_answers, prior_dims = cl.prior_answers_from_study(Path(args.study_logs), m.nick)
    prior_prof, prior_q = cl.profile({k: list(v) for k, v in prior_answers.items()}, prior_dims)
    n_prior = int(pd.Series([len(v) for v in prior_answers.values()]).median())
    print(f"{m.nick}: {len(args.invitations)} invitations x {args.chains} chains x {args.rounds} rounds; prior pooled over {n_prior} answers/item")

    # ---- loop ----
    versions_dir = out / "versions"
    steps: list[dict] = []
    for inv in args.invitations:
        for k in range(args.chain_start, args.chain_start + args.chains):
            vdir = versions_dir / inv / f"chain{k}"
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / "round0.md").write_text(start_text + "\n", encoding="utf-8")
            current = start_text
            kinds, justs = ["start"], [""]
            for t in range(args.rounds + 1):
                vpath = vdir / f"round{t}.md"
                ldir = out / "logs" / inv / f"chain{k}" / f"round{t}"
                if "loop" in args.stages or done(ldir / "vb") is not None:
                    label = json.dumps({"model": m.nick, "constitution": f"loop:{inv}/c{k}/r{t}", "chain": k, "round": t, "repeat": 0,
                                        "invitation": inv})
                    vb_log = run(valuebench_task(constitution=str(vpath), questionnaires=args.questionnaires, split=args.split, label=label,
                                                 reasoning=reasoning), ldir / "vb", m.inspect_name, roles, args, m.model_args, epochs=args.vb_epochs)
                else:
                    break
                answers, dims = cl.vb_answers(vb_log)
                prof, q = cl.profile(answers, dims)
                test = ea.permutation_w1({i: pd.Series(v).to_numpy() for i, v in answers.items()}, prior_answers, n_perm=200)
                steps.append({"invitation": inv, "chain": k, "round": t, "kind": kinds[-1], "words": len(current.split()),
                              "principles": len(parse_principles(current)),
                              "change_from_previous": change_ratio(Path(vdir / f"round{t - 1}.md").read_text().strip(), current) if t else 0.0,
                              "change_from_start": change_ratio(start_text, current), "w1_from_prior": test["observed_w1"],
                              "null_mean": test["null_mean"], "excess": test["excess"], "p_value": test["p_value"],
                              "justification": justs[-1][:400]})
                if t == args.rounds:
                    break
                evidence = cl.evidence_block(prof, prior_prof, q, n_prior, args.vb_epochs)
                invitation = evidence + ("" if inv == "reflective" else prompts[inv])
                if inv == "reflective":
                    from objective_editing import prompts as oe_prompts
                    invitation = evidence + oe_prompts.REFLECTIVE_INVITATION
                elog = run(chain_task(constitution=str(vpath), rounds=1, reasoning=reasoning, invitation=invitation, invitation_id=f"loop:{inv}",
                                      label=json.dumps({"invitation": inv, "chain": k, "round": t})),
                           ldir / "edit", m.inspect_name, None, args, m.model_args)
                st = elog.samples[0].store
                current = st["ChainStore:versions"][-1]
                kinds.append(st["ChainStore:kinds"][-1]); justs.append(st["ChainStore:justifications"][-1])
                (vdir / f"round{t + 1}.md").write_text(current + "\n", encoding="utf-8")
                print(f"[loop] {inv} chain {k} round {t + 1}: {kinds[-1]} ({change_ratio(start_text, current):.0%} from start)")
    res = out / "results"; res.mkdir(parents=True, exist_ok=True)
    steps_df = pd.DataFrame(steps)
    steps_df.to_csv(res / "steps.csv", index=False)

    # ---- eigen ----
    scores = {}
    if "eigen" in args.stages and not steps_df.empty:
        scenarios = load_scenarios(limit=args.scenarios_limit)
        responses, anchors = {}, []
        ents = {"prior": "none", "start": str(args.constitution)}
        for _, r in steps_df.iterrows():
            if r["round"] > 0:
                ents[f"{r['invitation']}|c{int(r['chain'])}|r{int(r['round'])}"] = str(versions_dir / r["invitation"] / f"chain{int(r['chain'])}" / f"round{int(r['round'])}.md")
        for ent, path in ents.items():
            log = run(eigen_responses_task(path, entity=ent, scenarios_limit=args.scenarios_limit, reasoning=reasoning),
                      out / "logs" / "eigen" / "responses" / safe_name(ent), m.inspect_name, None, args, m.model_args)
            responses[ent] = responses_from_log(log)
        anchors = ["prior", "start"]
        pairs = make_pairs(responses, anchors, scenarios, random_pairs_per_scenario=args.random_pairs)
        pf = res / "eigen_pairs.jsonl"; pf.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
        print(f"[eigen] {len(ents)} entities, judging {len(pairs)} pairs")
        ref = args.reference or str(args.constitution)
        jlog = run(eigen_judge_task(str(pf), ref), out / "logs" / "eigen" / "judgments", args.judge, roles, args)
        names = build_evaluations(jlog, "judge", res / "eigen_evaluations.jsonl")
        scores = fit_scores(jlog, names, "judge")
        (res / "eigen_scores.json").write_text(json.dumps(scores, indent=1))
    elif (res / "eigen_scores.json").exists():
        scores = json.loads((res / "eigen_scores.json").read_text())
    if scores:
        steps_df["eigen_score"] = [scores.get("start") if r["round"] == 0 else scores.get(f"{r['invitation']}|c{int(r['chain'])}|r{int(r['round'])}")
                                   for _, r in steps_df.iterrows()]
        steps_df.to_csv(res / "steps.csv", index=False)

    # ---- report ----
    if "report" in args.stages and not steps_df.empty:
        cl.plot_loop_editing(steps_df, res / "figures" / "loop_editing.png")
        cl.plot_loop_scores(steps_df, res / "figures" / "loop_scores.png", prior_eigen=scores.get("prior") if scores else None)
        pd.set_option("display.width", 220)
        cols = ["invitation", "chain", "round", "kind", "change_from_start", "excess", "p_value"] + (["eigen_score"] if scores else [])
        print(steps_df[cols].round(3).to_string(index=False))
        print(f"\nwritten to {res}")


if __name__ == "__main__":
    main()
