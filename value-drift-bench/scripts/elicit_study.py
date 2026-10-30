#!/usr/bin/env python3
"""Elicitation study: how much do elicited constitutions vary, and how much of that is the prompt?

    python scripts/elicit_study.py --models qwen3.5-9b claude-sonnet-5 \\
        --judge openrouter/deepseek/deepseek-v4-flash-0731 --judge2 openrouter/google/gemini-3.8-flash \\
        --variants oat --replicates 5 --out runs/elicit-pilot --dry-run

Design: prompt variants (valuedrift.elicit_variants; "oat" = six one-factor-at-a-time cells,
"all" = the 3 x 3 x 2 factorial) x models x sampled replicates at --temperature.

Stages (each skipped if its log directory already holds a successful run):
  elicit     every (model, variant) cell, --replicates samples; texts to <out>/elicitations/,
             manifest to <out>/results/elicitations.jsonl (rejected refusals included, flagged)
  code       every valid constitution, principle by principle, by --judge (and --judge2 for
             agreement) -> results/coding.csv
  axes       every valid constitution rated by --judge on two 1-7 axes (self-model vs ethics
             code; warm vs policy-like) -> results/text_axes.csv
  measure    ValueBench with each valid constitution as the system prompt; the prior
             --prior-repeats times (default 2; 10 is recommended, see README) and the base
             cell's first replicate twice (noise floor)
  analysis   results/: profiles.csv, variance_decomposition.csv, inertness.csv,
             priming_valuebench.csv, priming_coding.csv, text_distances.json, echo.csv, shifts.csv,
             figures/umap_elicitations.png, figures/shift_heatmap.png, figures/text_axes.png, and the
             noise-reduced variance_decomposition_coarse.csv, shifts_coarse.csv,
             figures/coarser_umap_elicitations.png, figures/shift_heatmap_coarse.png, and
             figures/all_model_umap.png (one arrow per model, prior -> all its constitutions pooled,
             annotated with the excess over the permutation no-effect floor from prior_shift_permutation.csv),
             figures/all_model_pca.png (the same arrows on a PCA of the states' value profiles, a linear
             layout whose distances and axes are interpretable; loadings in all_model_pca_loadings.csv)

Judges must come from families not under study. Labels written into every measurement log:
  {"model", "constitution", "chain": null, "round": null, "repeat", "variant", "replicate"}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all import run  # noqa: E402  (same eval runner, resume and dry-run semantics)

import pandas as pd  # noqa: E402

from valuedrift import elicit_analysis as ea  # noqa: E402
from valuedrift.analysis import dimension_profiles  # noqa: E402
from valuedrift.coding import coding_profiles, coding_table, coding_task  # noqa: E402
from valuedrift.elicit_variants import (BASE, Variant, elicit_variants_task, export_elicitations,  # noqa: E402
                                        merge_manifest, read_manifest, variants, write_manifest)
from valuedrift.registry import ROOT, load_models, safe_name  # noqa: E402
from valuedrift.text_axes import axes_table, axes_task, plot_text_axes  # noqa: E402
from valuedrift.valuebench import valuebench_task  # noqa: E402


def label(model: str, constitution: str, repeat: int = 0, variant: str | None = None, replicate: int | None = None) -> str:
    return json.dumps({"model": model, "constitution": constitution, "chain": None, "round": None,
                       "repeat": repeat, "variant": variant, "replicate": replicate})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="nicks from models.yaml (default: all)")
    ap.add_argument("--variants", default="oat", help="oat | all | comma-separated ids like actual-set_a-ai")
    ap.add_argument("--replicates", type=int, default=5, help="sampled constitutions per (model, variant)")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature for elicitation")
    ap.add_argument("--judge", required=True, help="principle coder and ValueBench evaluator")
    ap.add_argument("--judge2", help="second principle coder, for agreement (a different family)")
    ap.add_argument("--judge-reasoning", default="none", help="reasoning setting for --judge on coding calls")
    ap.add_argument("--judge2-reasoning", default="minimal", help="reasoning setting for --judge2 on coding calls (some endpoints refuse none)")
    ap.add_argument("--questionnaires", default="PVQ40,MFT23,LVI,HEXACO")
    ap.add_argument("--split", default="measure", choices=["all", "select", "measure"])
    ap.add_argument("--vb-epochs", type=int, default=1)
    ap.add_argument("--prior-repeats", type=int, default=2,
                    help="ValueBench runs of the prior per model; the prior is every shift's reference, so more runs sharpen every result")
    ap.add_argument("--stages", nargs="*", default=["elicit", "code", "axes", "measure", "analysis"])
    ap.add_argument("--reasoning", default="none", help="subject reasoning; a model's `reasoning` in models.yaml overrides it")
    ap.add_argument("--run-label", default="", help="free text recorded in every log and manifest entry of this invocation")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-connections", type=int, default=10)
    ap.add_argument("--display", default="plain")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models-file", default=str(ROOT / "models.yaml"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "elicit-study"))
    args = ap.parse_args()

    out = Path(args.out)
    logs, elicited, results = out / "logs", out / "elicitations", out / "results"
    manifest_path = results / "elicitations.jsonl"
    judge_roles = {"judge": args.judge} | ({"judge2": args.judge2} if args.judge2 else {})
    models = load_models(Path(args.models_file), nicks=args.models)
    cells = variants(args.variants)
    print(f"{len(models)} models x {len(cells)} variants x {args.replicates} replicates; stages {args.stages}")

    # ---- elicit ----
    manifest: list[dict] = read_manifest(manifest_path) if manifest_path.exists() else []
    if "elicit" in args.stages:
        new: list[dict] = []
        for m in models:
            for v in cells:
                print(f"[elicit] {m.nick} / {v.id}")
                log = run(elicit_variants_task(m.nick, v.id, args.replicates, m.reasoning_or(args.reasoning), args.temperature,
                                               label=args.run_label),
                          logs / "elicit" / safe_name(m.nick) / safe_name(v.id), m.inspect_name, None, args, m.model_args)
                if log is None:
                    continue
                if log.status != "success":
                    print(f"  skipped: eval {log.status}; rerun to retry")
                    continue
                new += export_elicitations(log, elicited, m.nick)
        if new:
            manifest = merge_manifest(manifest, new)
            write_manifest(manifest, manifest_path)
            n_valid = sum(e["valid"] for e in manifest)
            runs = sorted({e["run_id"] for e in manifest})
            print(f"[elicit] {n_valid}/{len(manifest)} constitutions valid from {len(runs)} run(s); manifest {manifest_path}")
    valid = [e for e in manifest if e["valid"]]
    if args.dry_run and not manifest:
        print(f"  (code and measure plan the {len(models) * len(cells) * args.replicates} elicited constitutions once they exist)")

    # ---- code ----
    coding_log = None
    if "code" in args.stages and valid:
        print(f"[code] {len(valid)} constitutions, judges {list(judge_roles)}")
        jr = ",".join([args.judge_reasoning] + ([args.judge2_reasoning] if args.judge2 else []))
        coding_log = run(coding_task(str(manifest_path), ",".join(judge_roles), judge_reasoning=jr), logs / "coding", args.judge,
                         judge_roles, args)
        if coding_log is not None and coding_log.status == "success":
            results.mkdir(parents=True, exist_ok=True)
            coding_table(coding_log).to_csv(results / "coding.csv", index=False)

    # ---- axes ----
    if "axes" in args.stages and valid:
        print(f"[axes] {len(valid)} constitutions rated by {args.judge}")
        alog = run(axes_task(str(manifest_path)), logs / "axes", args.judge, {"judge": args.judge}, args)
        if alog is not None and alog.status == "success":
            results.mkdir(parents=True, exist_ok=True)
            axes_table(alog).to_csv(results / "text_axes.csv", index=False)

    # ---- measure ----
    if "measure" in args.stages:
        def vb(m, c_name, text_path, log_dir, repeat=0, variant=None, replicate=None):
            return run(valuebench_task(constitution=text_path, questionnaires=args.questionnaires, split=args.split,
                                       label=label(m.nick, c_name, repeat, variant, replicate), reasoning=m.reasoning_or(args.reasoning)),
                       log_dir, m.inspect_name, {"judge": args.judge}, args, m.model_args, epochs=args.vb_epochs)
        base_id = Variant(*BASE).id
        for m in models:
            for repeat in range(args.prior_repeats):
                print(f"[measure] {m.nick} / prior (repeat {repeat})")
                vb(m, "none", "none", logs / "valuebench" / safe_name(m.nick) / "none" / f"repeat{repeat}", repeat)
            for e in [x for x in valid if x["model"] == m.nick]:
                c_name = f"elicited:{e['variant']}/r{e['replicate']}"
                d = logs / "valuebench" / safe_name(m.nick) / safe_name(e["variant"]) / f"r{e['replicate']}"
                print(f"[measure] {m.nick} / {c_name}")
                vb(m, c_name, e["path"], d / "repeat0", 0, e["variant"], e["replicate"])
                if e["variant"] == base_id and e["replicate"] == min(x["replicate"] for x in valid if x["model"] == m.nick and x["variant"] == base_id):
                    vb(m, c_name, e["path"], d / "repeat1", 1, e["variant"], e["replicate"])   # noise floor

    # ---- analysis ----
    if "analysis" in args.stages and not args.dry_run:
        results.mkdir(parents=True, exist_ok=True)
        if manifest:
            ea.echo_table(manifest).to_csv(results / "echo.csv", index=False)
        df = ea.load_study(logs / "valuebench")
        if df.empty:
            print("[analysis] no ValueBench results yet")
        else:
            meta = ea.entity_meta(df)
            prof = dimension_profiles(df)
            prof.to_csv(results / "profiles.csv")
            variance = ea.variance_decomposition(prof, meta)
            variance.to_csv(results / "variance_decomposition.csv", index=False)
            ea.inertness(df, meta).to_csv(results / "inertness.csv", index=False)
            ea.priming_test(prof, meta).to_csv(results / "priming_valuebench.csv", index=False)
            ea.plot_umap(df, results / "figures" / "umap_elicitations.png", meta)
            shifts = ea.shift_table(prof, meta, variance, ea.dimension_questionnaires(df))
            shifts.to_csv(results / "shifts.csv", index=False)
            ea.plot_shift_heatmap(shifts, results / "figures" / "shift_heatmap.png")
            # noise-reduced views: replicates pooled per prompt cell; facets rolled up to domains
            pooled = ea.pool_cells(df)
            ea.plot_umap(pooled, results / "figures" / "coarser_umap_elicitations.png", ea.entity_meta(pooled),
                         title="Elicited constitutions, replicates pooled per prompt cell (UMAP of W1 distances)",
                         point_label="prompt cell (5 constitutions pooled)")
            groups = {m.nick: m.group for m in load_models(Path(args.models_file))}
            pshift = ea.prior_shift_table(df)
            pshift.to_csv(results / "prior_shift_permutation.csv", index=False)
            ea.plot_model_arrows(ea.pool_models(df), groups, results / "figures" / "all_model_umap.png", shifts=pshift)
            _, loadings, _ = ea.plot_model_arrows_pca(ea.pool_models(df), groups, results / "figures" / "all_model_pca.png", shifts=pshift)
            loadings.to_csv(results / "all_model_pca_loadings.csv")
            cprof = ea.coarse_profiles(df)
            cvar = ea.variance_decomposition(cprof, meta)
            cvar.to_csv(results / "variance_decomposition_coarse.csv", index=False)
            cshifts = ea.shift_table(cprof, meta, cvar, ea.coarse_questionnaires(df))
            cshifts.to_csv(results / "shifts_coarse.csv", index=False)
            ea.plot_shift_heatmap(cshifts, results / "figures" / "shift_heatmap_coarse.png",
                                  title="Shift from the prior under elicited constitutions, coarse dimensions")
        if (results / "coding.csv").exists():
            table = pd.read_csv(results / "coding.csv")
            cprof = coding_profiles(table)
            cmeta = ea.add_factors(table.drop_duplicates("entity").set_index("entity")[["model", "variant", "replicate"]]
                                   .assign(constitution=lambda t: "elicited:" + t["variant"], repeat=0))
            ea.variance_decomposition(cprof, cmeta).to_csv(results / "variance_decomposition_coding.csv", index=False)
            ea.priming_test(cprof, cmeta).to_csv(results / "priming_coding.csv", index=False)
        if (results / "text_axes.csv").exists():
            groups = {m.nick: m.group for m in load_models(Path(args.models_file))}
            plot_text_axes(pd.read_csv(results / "text_axes.csv"), groups, results / "figures" / "text_axes.png")
        if valid:
            texts = {f"{e['model']}|{e['variant']}|r{e['replicate']}": Path(e["path"]).read_text(encoding="utf-8") for e in valid}
            tmeta = ea.add_factors(pd.DataFrame(valid).assign(entity=list(texts), repeat=0,
                                                              constitution=lambda t: "elicited:" + t["variant"])
                                   .set_index("entity")[["model", "variant", "replicate", "repeat", "constitution"]])
            (results / "text_distances.json").write_text(json.dumps(ea.text_distance_summary(texts, tmeta), indent=1))
        print(f"[analysis] wrote tables to {results}")


if __name__ == "__main__":
    main()
