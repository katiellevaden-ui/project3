#!/usr/bin/env python3
"""Run the whole pipeline for a set of models and every constitution in constitutions/.

    python scripts/run_all.py --models qwen3.5-9b claude-sonnet-5 --judge openrouter/deepseek/deepseek-v4-flash-0731 \\
        --rounds 6 --chains 3 --vb-epochs 3 --checkpoints 0 3 6 --dry-run

Stages (each skipped if its log directory already holds a successful run, so reruns resume):
  elicit      constitutions/elicited/<nick>.md                         (per model)
  baseline    ValueBench for the prior and for every constitution, twice (noise floor)
  chain       reflective recursive editing, `--chains` independent chains, exported to versions/
  measure     ValueBench for every version of every chain
  eigen       EigenBench responses for priors, baselines and checkpoint versions; one external-judge
              batch; EigenTrust scores to results/eigen_scores.json
  analysis    tables and figures under results/

Labels written into every measurement log identify the state:
  {"model", "constitution", "chain", "round", "repeat"}  (round null = baseline; constitution "none" = prior)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.log import list_eval_logs, read_eval_log

from valuedrift import analysis
from valuedrift.chain import chain_task, export_versions
from valuedrift.eigen import (build_evaluations, eigen_judge_task, eigen_responses_task, fit_scores,
                              load_scenarios, make_pairs, responses_from_log)
from valuedrift.elicit import elicit_task
from valuedrift.registry import ROOT, load_constitutions, load_models, safe_name
from valuedrift.valuebench import valuebench_task

LOGS, VERSIONS, RESULTS = ROOT / "logs", ROOT / "versions", ROOT / "results"
load_dotenv(ROOT / ".env")   # Inspect's Python eval() does not read .env; the CLI does


def label(model, constitution, chain=None, round=None, repeat=0) -> str:
    return json.dumps({"model": model, "constitution": constitution, "chain": chain, "round": round, "repeat": repeat})


def done(log_dir: Path):
    """The successful log in a directory, or None."""
    if not log_dir.exists():
        return None
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name, header_only=True)
        if log.status == "success":
            return read_eval_log(info.name)
    return None


def run(task, log_dir: Path, model, roles: dict | None, args, model_args: dict | None = None, **kw):
    """Run one eval unless already done. Returns the log."""
    existing = done(log_dir)
    if existing is not None:
        return existing
    if args.dry_run:
        print(f"  would run {task.name:>22} -> {log_dir}  model={model}")
        return None
    [log] = inspect_eval(task, model=model, model_args=model_args or {}, model_roles=roles or {},
                         log_dir=str(log_dir), timeout=args.timeout, max_retries=3,
                         max_connections=args.max_connections, display=args.display, **kw)
    if log.status != "success":
        print(f"  FAILED {log_dir}: {log.error}", file=sys.stderr)
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="nicks from models.yaml (default: all)")
    ap.add_argument("--constitutions", nargs="*", help="constitution names (default: every file in constitutions/)")
    ap.add_argument("--judge", required=True, help="ValueBench evaluator, e.g. openrouter/deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--eigen-judge", help="EigenBench judge (default: --judge)")
    ap.add_argument("--reference", default="c0_general_assistant", help="constitution every EigenBench pair is judged against")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--vb-epochs", type=int, default=1, help="answers per ValueBench item (>=2 enables item distributions)")
    ap.add_argument("--questionnaires", default="PVQ40,MFT23,LVI,HEXACO")
    ap.add_argument("--split", default="measure", choices=["all", "select", "measure"])
    ap.add_argument("--checkpoints", nargs="*", type=int, default=[0, 3, 6], help="rounds measured with EigenBench")
    ap.add_argument("--scenarios-limit", type=int, default=30, help="EigenBench scenarios per entity")
    ap.add_argument("--random-pairs", type=int, default=3)
    ap.add_argument("--stages", nargs="*", default=["elicit", "baseline", "chain", "measure", "eigen", "analysis"])
    ap.add_argument("--no-elicited", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-connections", type=int, default=10)
    ap.add_argument("--display", default="plain")
    ap.add_argument("--reasoning", default="none", help="model-under-study reasoning: none (default) | default | low | medium | high")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models-file", default=str(ROOT / "models.yaml"))
    ap.add_argument("--out", default=str(ROOT), help="root for logs/, versions/ and results/")
    args = ap.parse_args()
    global LOGS, VERSIONS, RESULTS
    LOGS, VERSIONS, RESULTS = Path(args.out) / "logs", Path(args.out) / "versions", Path(args.out) / "results"
    eigen_judge = args.eigen_judge or args.judge
    judge_roles = {"judge": args.judge}

    models = load_models(Path(args.models_file), nicks=args.models)
    print(f"{len(models)} models; stages {args.stages}; constitutions dir {ROOT / 'constitutions'}")

    # ---- elicit ----
    if "elicit" in args.stages and not args.no_elicited:
        for m in models:
            print(f"[elicit] {m.nick}")
            run(elicit_task(m.nick, reasoning=m.reasoning_or(args.reasoning)), LOGS / "elicit" / safe_name(m.nick), m.inspect_name, None, args, m.model_args)

    plan = []        # (model, constitution) pairs
    for m in models:
        for c in load_constitutions(None if args.no_elicited else m.nick):
            if args.constitutions and c.name not in args.constitutions and not c.is_prior and c.source != "elicited":
                continue
            plan.append((m, c))

    def vb(m, c_name, text_path, chain=None, rnd=None, repeat=0, epochs=1):
        d = LOGS / "valuebench" / safe_name(m.nick) / safe_name(c_name) / (
            f"baseline{repeat}" if rnd is None else f"chain{chain}_round{rnd}")
        return run(valuebench_task(constitution=text_path or "none", questionnaires=args.questionnaires, split=args.split,
                                   label=label(m.nick, c_name, chain, rnd, repeat), reasoning=m.reasoning_or(args.reasoning)),
                   d, m.inspect_name, judge_roles, args, m.model_args, epochs=epochs)

    def const_path(m, c) -> str:
        if c.is_prior:
            return "none"
        if c.source == "elicited":
            return str(ROOT / "constitutions" / "elicited" / f"{safe_name(m.nick)}.md")
        return str(ROOT / "constitutions" / f"{c.name}.md")

    # ---- baseline ----
    if "baseline" in args.stages:
        for m, c in plan:
            for repeat in (0, 1):                               # twice: the noise floor
                print(f"[baseline] {m.nick} / {c.name} (repeat {repeat})")
                vb(m, c.name, const_path(m, c), repeat=repeat, epochs=args.vb_epochs)

    # ---- chain ----
    version_index: dict = {}
    if "chain" in args.stages or "measure" in args.stages or "eigen" in args.stages:
        for m, c in plan:
            if c.is_prior:
                continue
            print(f"[chain] {m.nick} / {c.name}")
            log = run(chain_task(constitution=const_path(m, c), rounds=args.rounds, label=label(m.nick, c.name),
                                 reasoning=m.reasoning_or(args.reasoning)),
                      LOGS / "chains" / safe_name(m.nick) / safe_name(c.name), m.inspect_name, None, args,
                      m.model_args, epochs=args.chains)
            if log is not None and log.status == "success":
                version_index[(m.nick, c.name)] = export_versions(log, VERSIONS, m.nick, c.name)

    # ---- measure ----
    if "measure" in args.stages:
        for (nick, c_name), entries in version_index.items():
            m = next(x for x in models if x.nick == nick)
            for v in entries:
                if v["round"] == 0:
                    continue                                    # round 0 == the baseline measurement
                print(f"[measure] {nick} / {c_name} chain {v['chain']} round {v['round']}")
                vb(m, c_name, v["path"], chain=v["chain"], rnd=v["round"], epochs=args.vb_epochs)

    # ---- eigen ----
    if "eigen" in args.stages:
        scenarios = load_scenarios(limit=args.scenarios_limit)
        responses, anchors = {}, []
        for m, c in plan:
            ent = f"{m.nick}|{c.name}|base"
            log = run(eigen_responses_task(const_path(m, c), entity=ent, scenarios_limit=args.scenarios_limit, reasoning=m.reasoning_or(args.reasoning)),
                      LOGS / "eigen" / "responses" / safe_name(ent), m.inspect_name, None, args, m.model_args)
            if log is not None:
                responses[ent] = responses_from_log(log)
                anchors.append(ent)                             # priors and baselines anchor the comparison graph
        for (nick, c_name), entries in version_index.items():
            m = next(x for x in models if x.nick == nick)
            for v in entries:
                if v["round"] not in args.checkpoints or v["round"] == 0:
                    continue
                ent = f"{nick}|{c_name}|c{v['chain']}r{v['round']}"
                log = run(eigen_responses_task(v["path"], entity=ent, scenarios_limit=args.scenarios_limit, reasoning=m.reasoning_or(args.reasoning)),
                          LOGS / "eigen" / "responses" / safe_name(ent), m.inspect_name, None, args, m.model_args)
                if log is not None:
                    responses[ent] = responses_from_log(log)
        if responses and not args.dry_run:
            pairs = make_pairs(responses, anchors, scenarios, random_pairs_per_scenario=args.random_pairs)
            pairs_file = RESULTS / "eigen_pairs.jsonl"
            pairs_file.parent.mkdir(parents=True, exist_ok=True)
            pairs_file.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
            print(f"[eigen] judging {len(pairs)} pairs with {eigen_judge}")
            ref = str(ROOT / "constitutions" / f"{args.reference}.md")
            jlog = run(eigen_judge_task(str(pairs_file), ref), LOGS / "eigen" / "judgments", eigen_judge,
                       {"judge": eigen_judge}, args)      # main model unused; the solver calls the judge role
            if jlog is not None:
                names = build_evaluations(jlog, "judge", RESULTS / "eigen_evaluations.jsonl")
                scores = fit_scores(jlog, names, "judge")
                (RESULTS / "eigen_scores.json").write_text(json.dumps(scores, indent=1))
                analysis.plot_eigen_scores(scores, RESULTS / "figures" / "eigen_scores.png")
        elif args.dry_run:
            print(f"  would judge pairs for {len(plan)} baseline entities + checkpoint versions")

    # ---- analysis ----
    if "analysis" in args.stages and not args.dry_run:
        df = analysis.load_valuebench(LOGS / "valuebench")
        if df.empty:
            print("[analysis] no ValueBench results yet"); return
        RESULTS.mkdir(exist_ok=True)
        analysis.dimension_profiles(df).to_csv(RESULTS / "dimension_profiles.csv")
        analysis.drift_table(df).to_csv(RESULTS / "drift.csv", index=False)
        analysis.noise_floor(df).to_csv(RESULTS / "noise_floor.csv", index=False)
        analysis.plot_trajectories(df, RESULTS / "figures" / "umap_trajectories.png")
        print(f"[analysis] wrote tables and figures to {RESULTS}")


if __name__ == "__main__":
    main()
