"""Recursive constitution editing with training, measured the way the inference study measured it.

Round 0: M_0 elicits C_0 (unless the config supplies one) and is measured with no system
prompt and with C_0 in its prompt. Round 1 (config "train_initial", default true) trains M_0
on the unedited C_0 -> M_1, so that every later edit is driven by internalised behaviour and
the fixed-C_0 control shares this checkpoint. Each later round t (state machine, resumable):
    measure      serve M_t; ValueBench with no system prompt, `measure_repeats` runs; also M_0
                 with C_t in the prompt (the inference comparator)
    edit         serve M_t; show it its trained measurement against M_0's prior and invite it
                 to revise C_t under the study's gen10 invitation; the reply is C_{t+1}
                 (config "edit": false keeps C_t: the fixed-constitution control)
    preferences  M_t answers the prompt bank with C_{t+1} in its system prompt (teacher = M_t
                 itself by default, "same setup as inference") and without it (student); the
                 chosen/rejected gap is logged (length ratio, near-identical share)
    dpo          recursive_oct's full-parameter DPO; the post-DPO checkpoint is measured
                 (config "measure_post_dpo", default true) to attribute change to DPO vs SFT
    introspection, sft           recursive_oct's stages -> M_{t+1}
    elicit       serve M_{t+1}; it writes its constitution afresh (diagnostic only)
`max_rounds` counts training rounds, including the initial one.
Config "train": false runs the inference-only arm on the same served model and C_0: no
training, the constitution stays in the prompt, and each round's evidence is M_0 with C_t in
its prompt against M_0's prior (the inference study's protocol); `max_rounds` then counts edits.
The preference gap is judged, not counted: a blind judge compares chosen and rejected answers
on a subsample against C_{t+1} (both orders); if the chosen side's win rate falls below
"dpo_skip_below_win_rate" (default 0.6) DPO is skipped and M_t goes straight to introspective
SFT, so the SFT mechanism is still tested when distillation saturates. Nothing aborts.

The inference study's modules supply elicitation, the invitation, the evidence block and the
measurement tasks (valuedrift.*); recursive_oct supplies generation and training. Every
Inspect task runs against the served checkpoint through ServedModel.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import write_json
from .serving import ServedModel

TERMINAL = {"SELF_DECLARED_CONVERGENCE", "ROUND_LIMIT", "EDITING_FAILURE", "TRAINING_FAILURE", "BUDGET_LIMIT"}


def _inspect_eval(task, model, log_dir, bench, **kw):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import list_eval_logs, read_eval_log
    log_dir = Path(log_dir)
    if log_dir.exists():
        for info in list_eval_logs(str(log_dir)):
            if read_eval_log(info.name, header_only=True).status == "success":
                return read_eval_log(info.name)
    [log] = inspect_eval(task, model=model, model_roles={"judge": bench["judge"]}, log_dir=str(log_dir),
                         max_connections=bench.get("max_connections", 16), display=bench.get("display", "plain"),
                         max_retries=3, timeout=bench.get("timeout", 900), **kw)
    if log.status != "success":
        raise RuntimeError(f"{task.name} failed: {log.error}")
    return log


def _label(tag, constitution, repeat, round_, checkpoint, prompted=None):
    return json.dumps({"model": tag, "constitution": constitution, "chain": None, "round": round_, "repeat": repeat,
                       "checkpoint": str(checkpoint), "prompted_with": prompted})


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def measure(served: ServedModel, constitution_path: str | None, out_dir: Path, bench: dict, tag: str, round_: int, checkpoint):
    """ValueBench runs of the served model, with or without a constitution in the system prompt."""
    from valuedrift.valuebench import valuebench_task
    logs = []
    for k in range(bench.get("measure_repeats", 2)):
        task = valuebench_task(constitution=constitution_path or "none", questionnaires=bench["questionnaires"], split=bench.get("split", "measure"),
                               label=_label(tag, "none" if constitution_path is None else f"prompted:{Path(constitution_path).name}", k, round_, checkpoint,
                                            None if constitution_path is None else str(constitution_path)),
                               reasoning=bench.get("reasoning", "default"))
        logs.append(_inspect_eval(task, served.model, out_dir / f"repeat{k}", bench, epochs=bench.get("vb_epochs", 3)))
    return logs


def answers_from(logs) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    from valuedrift.closed_loop import vb_answers
    pooled: dict[str, list[float]] = {}; dims = {}
    for log in logs:
        a, d = vb_answers(log); dims.update(d)
        for i, v in a.items():
            pooled.setdefault(i, []).extend(v)
    return {i: np.array(v) for i, v in pooled.items()}, dims


def elicit(served: ServedModel, out_dir: Path, bench: dict, tag: str) -> Path:
    from valuedrift.elicit import elicit_task
    _inspect_eval(elicit_task(tag, out_dir=str(out_dir), reasoning=bench.get("reasoning", "default")), served.model, out_dir / "log", bench)
    accepted = out_dir / f"{tag}.md"
    if not accepted.exists():
        raise RuntimeError(f"{tag} did not produce a valid constitution (see {out_dir})")
    return accepted


def edit(served: ServedModel, constitution_path: Path, evidence: str, out_dir: Path, bench: dict) -> dict:
    from valuedrift.chain import chain_task
    inv = evidence + bench["invitation_text"]
    log = _inspect_eval(chain_task(constitution=str(constitution_path), rounds=1, reasoning=bench.get("reasoning", "default"), invitation=inv,
                                   invitation_id=f"bench:{bench.get('invitation_id', 'gen10')}", label="bench-edit"),
                        served.model, out_dir / "log", bench)
    st = log.samples[0].store
    kind = st["ChainStore:kinds"][-1]
    return {"status": "EDITED" if kind == "edit" else ("SELF_DECLARED_CONVERGENCE" if kind == "unchanged" else "EDITING_FAILURE"),
            "kind": kind, "text": st["ChainStore:versions"][-1], "justification": st["ChainStore:justifications"][-1],
            "error": st.get("ChainStore:error")}


def evidence_for(with_logs, prior_logs, bench: dict) -> str:
    from valuedrift.closed_loop import evidence_block, profile
    a_with, d_with = answers_from(with_logs); a_prior, d_prior = answers_from(prior_logs)
    p_with, q = profile({k: list(v) for k, v in a_with.items()}, d_with)
    p_prior, _ = profile({k: list(v) for k, v in a_prior.items()}, d_prior)
    n_prior = int(np.median([len(v) for v in a_prior.values()])); n_with = int(np.median([len(v) for v in a_with.values()]))
    return evidence_block(p_with, p_prior, q, n_prior, n_with)


def preference_gap(preferences_path: Path) -> dict:
    """Length-bias diagnostic only: mean chosen/rejected word-length ratio and the word change
    ratio (two samples at temperature > 0 nearly always differ, so this is not a gap measure)."""
    from objective_editing.metrics import change_ratio
    rows = [json.loads(l) for l in Path(preferences_path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("chosen") and r.get("rejected")]
    if not rows:
        return {"pairs": 0, "length_ratio": None, "mean_change_ratio": None}
    ratios = [len(r["chosen"].split()) / max(len(r["rejected"].split()), 1) for r in rows]
    return {"pairs": len(rows), "length_ratio": float(np.mean(ratios)),
            "mean_change_ratio": float(np.mean([change_ratio(r["rejected"], r["chosen"]) for r in rows]))}


def preference_margin(preferences_path: Path, constitution_path: Path, out_dir: Path, bench: dict, seed: int = 0) -> dict:
    """The gap that matters: a blind judge picks, for a subsample of pairs, which answer better
    follows C_{t+1}, in both orders. Returns the chosen side's win rate (ties count half), the
    tie rate and the number of judged pairs. Uses the EigenBench pair judge unchanged."""
    import random
    from valuedrift.eigen import eigen_judge_task
    rows = [json.loads(l) for l in Path(preferences_path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("chosen") and r.get("rejected")]
    if not rows:
        return {"judged_pairs": 0, "chosen_win_rate": None, "tie_rate": None}
    sample = random.Random(seed).sample(rows, min(bench.get("margin_pairs", 50), len(rows)))
    pairs = []
    for i, r in enumerate(sample):
        for first, second in (("chosen", "rejected"), ("rejected", "chosen")):
            pairs.append({"id": f"{r.get('id', i)}|{first}|{second}", "scenario_index": i, "scenario": r["prompt"],
                          "eval1_name": first, "eval1_response": r[first], "eval2_name": second, "eval2_response": r[second]})
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = out_dir / "margin_pairs.jsonl"; pf.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
    log = _inspect_eval(eigen_judge_task(str(pf), str(constitution_path)), bench["judge"], out_dir / "log", bench)
    wins, ties, n = 0.0, 0, 0
    for smp in log.samples:
        choice = smp.scores["choice_record"].metadata.get("choice")
        if choice is None:
            continue
        n += 1
        if choice == 0:
            ties += 1; wins += 0.5
        elif (choice == 1) == (smp.metadata["eval1_name"] == "chosen"):
            wins += 1
    return {"judged_pairs": n, "chosen_win_rate": wins / n if n else None, "tie_rate": ties / n if n else None}


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def run_bench_trajectory(run_dir, config: dict, trainer, resume: bool = False, budget_ok=lambda: True) -> dict:
    root = Path(run_dir); root.mkdir(parents=True, exist_ok=True)
    bench, serve_cfg = dict(config["bench"]), config["serve"]
    if "invitation_path" in bench:
        bench["invitation_text"] = Path(bench["invitation_path"]).read_text()
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not resume:
            raise ValueError("Run exists; use explicit resume")
        if json.loads((root / "config.json").read_text()) != config:
            raise ValueError("Protocol changes require a separately labeled run")
        if state["status"] in TERMINAL - {"BUDGET_LIMIT", "TRAINING_FAILURE"}:
            return state
        state["status"] = "RUNNING"
    else:
        write_json(root / "config.json", config)
        state = {"status": "RUNNING", "phase": "elicit0", "completed_rounds": 0, "current_checkpoint": config["model"],
                 "current_constitution": None, "reviews": [], "failures": [], "started_epoch": time.time()}

    train = config.get("train", True)

    def save():
        state["updated_epoch"] = time.time(); write_json(state_path, state)

    def tag(t):
        return f"M{t:03d}"

    def serve(checkpoint, t):
        return ServedModel(checkpoint, serve_cfg, tag(t), log_dir=root / "serve-logs")

    save()
    while state["status"] == "RUNNING":
        if not budget_ok():
            state["status"] = "BUDGET_LIMIT"; save(); break
        phase, t = state["phase"], state["completed_rounds"]
        n = t + 1
        rd = root / f"round_{n:03d}"; rd.mkdir(exist_ok=True)
        try:
            if phase == "elicit0":
                c0 = root / "C_000.md"
                if config.get("constitution"):
                    c0.write_text(Path(config["constitution"]).read_text())
                else:
                    with serve(state["current_checkpoint"], 0) as served:
                        shutil.copy(elicit(served, root / "elicit_000", bench, tag(0)), c0)
                state["current_constitution"] = str(c0); state["phase"] = "measure"
            elif phase == "measure":
                with serve(state["current_checkpoint"], t) as served:
                    if t == 0 or train:
                        measure(served, None, root / "measure" / tag(t) / "none", bench, tag(t), t, state["current_checkpoint"])
                    if t == 0 or bench.get("measure_prompted_comparator", True):
                        # M_0 under the current constitution in its prompt: the inference-study comparator
                        if t == 0:
                            measure(served, state["current_constitution"], root / "measure" / tag(0) / f"prompted_C{t:03d}", bench, tag(0), t,
                                    state["current_checkpoint"])
                if t > 0 and bench.get("measure_prompted_comparator", True):
                    with serve(config["model"], 0) as served0:
                        measure(served0, state["current_constitution"], root / "measure" / tag(0) / f"prompted_C{t:03d}", bench, tag(0), t, config["model"])
                if t >= config.get("max_rounds", 3):
                    state["status"] = "ROUND_LIMIT"
                elif t == 0 and train and config.get("train_initial", True):
                    # round 1 trains on the unedited C_0: shared by the main arm and the fixed-C_0 control
                    submitted = root / f"C_{n:03d}.md"; submitted.write_text(Path(state["current_constitution"]).read_text())
                    write_json(rd / "review.json", {"status": "INITIAL", "kind": "initial", "text": submitted.read_text(), "justification": ""})
                    state["reviews"].append({"status": "INITIAL", "kind": "initial", "round": n})
                    state["submitted_constitution"] = str(submitted); state["phase"] = "preferences"
                else:
                    state["phase"] = "edit"
            elif phase == "edit":
                from inspect_ai.log import list_eval_logs, read_eval_log
                def load(d):
                    return [read_eval_log(i.name) for i in list_eval_logs(str(d), recursive=True) if read_eval_log(i.name, header_only=True).status == "success"]
                prior_logs = load(root / "measure" / tag(0) / "none")
                with_logs = load(root / "measure" / tag(0) / f"prompted_C{t:03d}") if (t == 0 or not train) else load(root / "measure" / tag(t) / "none")
                evidence = evidence_for(with_logs, prior_logs, bench)
                (rd / "evidence.txt").write_text(evidence)
                if (rd / "review.json").exists():
                    review = json.loads((rd / "review.json").read_text())
                elif not config.get("edit", True):
                    review = {"status": "EDITED", "kind": "fixed", "text": Path(state["current_constitution"]).read_text().strip(), "justification": ""}
                    write_json(rd / "review.json", review)
                else:
                    with serve(state["current_checkpoint"], t) as served:
                        review = edit(served, Path(state["current_constitution"]), evidence, rd / "edit", bench)
                    write_json(rd / "review.json", review)
                state["reviews"].append({k: v for k, v in review.items() if k != "text"})
                if review["status"] == "EDITED":
                    submitted = root / f"C_{n:03d}.md"; submitted.write_text(review["text"] + "\n")
                    state["submitted_constitution"] = str(submitted)
                    if train:
                        state["phase"] = "preferences"
                    else:                                             # inference-only arm: the edit is the whole round
                        state["current_constitution"] = state.pop("submitted_constitution")
                        state["completed_rounds"] = n; state["phase"] = "measure"
                else:
                    state["status"] = review["status"]
            elif phase == "preferences":
                trainer.preferences(state["current_checkpoint"], Path(state["submitted_constitution"]), rd / "preferences.jsonl")
                gap = preference_gap(rd / "preferences.jsonl")
                gap.update(preference_margin(rd / "preferences.jsonl", Path(state["submitted_constitution"]), rd / "margin", bench))
                write_json(rd / "preferences.gap.json", gap)
                state.setdefault("preference_gaps", []).append({"round": n, **gap})
                state["phase"] = "dpo"
            elif phase == "dpo":
                gap = state["preference_gaps"][-1]
                floor = config.get("dpo_skip_below_win_rate", 0.6)
                if gap.get("chosen_win_rate") is not None and gap["chosen_win_rate"] < floor:
                    # distillation has saturated: carry M_t forward and let introspective SFT act alone
                    state["dpo_checkpoint"] = state["current_checkpoint"]
                    state.setdefault("dpo_skipped", []).append({"round": n, "chosen_win_rate": gap["chosen_win_rate"]})
                    write_json(rd / "dpo_skipped.json", {"round": n, **gap, "threshold": floor})
                    state["phase"] = "introspection"
                else:
                    state["dpo_checkpoint"] = trainer.dpo(state["current_checkpoint"], rd / "preferences.jsonl", rd / "dpo")
                    state["phase"] = "measure_dpo" if bench.get("measure_post_dpo", True) else "introspection"
            elif phase == "measure_dpo":
                with serve(state["dpo_checkpoint"], n) as served:
                    served.model = served.model.replace(tag(n), f"{tag(n)}dpo") if serve_cfg.get("backend") == "scripted" else served.model
                    measure(served, None, root / "measure" / f"{tag(n)}_dpo" / "none", bench, f"{tag(n)}_dpo", n, state["dpo_checkpoint"])
                state["phase"] = "introspection"
            elif phase == "introspection":
                trainer.introspection(state["dpo_checkpoint"], rd / "introspection.jsonl", Path(state["submitted_constitution"]).read_text())
                state["phase"] = "sft"
            elif phase == "sft":
                state["next_checkpoint"] = trainer.sft(state["dpo_checkpoint"], rd / "introspection.jsonl", rd / "final")
                state["phase"] = "elicit"
            elif phase == "elicit":
                with serve(state["next_checkpoint"], n) as served:
                    try:
                        elicit(served, rd / "elicited_after_training", bench, tag(n))
                    except RuntimeError as exc:                     # diagnostic only: a refusal is recorded, not fatal
                        state["failures"].append({"phase": "elicit", "round": n, "message": str(exc)})
                state["current_checkpoint"] = state.pop("next_checkpoint")
                state["current_constitution"] = state.pop("submitted_constitution")
                state["completed_rounds"] = n
                state["phase"] = "measure"
            else:
                raise ValueError(f"Unknown phase: {phase}")
        except Exception as exc:
            state["failures"].append({"phase": phase, "time": time.time(), "type": type(exc).__name__, "message": str(exc)})
            state["status"] = "EDITING_FAILURE" if phase == "edit" else "TRAINING_FAILURE"
            save()
            import traceback
            with (rd / "failure.log").open("a") as f:
                traceback.print_exc(file=f)
        save()
    return state


# --------------------------------------------------------------------------
# Trainers
# --------------------------------------------------------------------------


class OCTTrainer:
    """recursive_oct's generation and training stages. teacher "self" means M_t itself answers
    the prompt bank with C_{t+1} in its system prompt (self-distillation, as in the inference
    study); a checkpoint path means a fixed external teacher (value-drift's original choice)."""

    def __init__(self, config: dict):
        from .train import read_jsonl
        self.config = config
        self.prompts = read_jsonl(config["train_prompts"])
        self.introspection_prompts = read_jsonl(config["introspection_prompts"])

    def preferences(self, checkpoint, constitution_path, output):
        from .generation import generate_preferences
        teacher = checkpoint if self.config.get("teacher", "self") == "self" else self.config["teacher"]
        return generate_preferences(checkpoint, teacher, Path(constitution_path).read_text(), self.prompts, output, self.config["generation"])

    def dpo(self, checkpoint, data, output):
        from .train import train_dpo
        return train_dpo(checkpoint, data, output, self.config["dpo"])["output_checkpoint"]

    def introspection(self, checkpoint, output, constitution_text):
        from .generation import generate_introspection
        return generate_introspection(checkpoint, self.introspection_prompts, output, self.config["introspection"], constitution=constitution_text)

    def sft(self, checkpoint, data, output):
        from .train import train_sft
        return train_sft(checkpoint, data, output, self.config["sft"])["output_checkpoint"]


class FakeTrainer:
    """No training: every stage writes a placeholder and returns the output path as the next
    checkpoint. For checking the loop's plumbing offline or on a pod before spending GPU time."""

    def preferences(self, checkpoint, constitution_path, output):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps({"id": "fake", "prompt": "", "chosen": "", "rejected": "", "note": "FakeTrainer"}) + "\n")

    def dpo(self, checkpoint, data, output):
        Path(output).mkdir(parents=True, exist_ok=True); (Path(output) / "FAKE_CHECKPOINT").write_text(str(checkpoint)); return str(output)

    def introspection(self, checkpoint, output, constitution_text):
        Path(output).write_text("{}\n")

    def sft(self, checkpoint, data, output):
        Path(output).mkdir(parents=True, exist_ok=True); (Path(output) / "FAKE_CHECKPOINT").write_text(str(checkpoint)); return str(output)


# --------------------------------------------------------------------------
# Report: distances and EigenBench across checkpoints
# --------------------------------------------------------------------------


def bench_report(run_dir, config: dict, eigen: bool = True) -> pd.DataFrame:
    """steps.csv: per checkpoint, W1 excess from M_0's prior (permutation floor), the prompted
    comparator's excess, text metrics of C_t; optional EigenBench of every checkpoint's
    no-prompt responses against C_0 and against C_t, in one batch per reference."""
    from inspect_ai.log import list_eval_logs, read_eval_log
    from objective_editing.data import parse_principles
    from objective_editing.metrics import change_ratio
    from valuedrift import elicit_analysis as ea
    root = Path(run_dir); bench = config["bench"]
    state = json.loads((root / "state.json").read_text())
    def load(d):
        return [read_eval_log(i.name) for i in list_eval_logs(str(d), recursive=True) if read_eval_log(i.name, header_only=True).status == "success"] if Path(d).exists() else []
    prior, _ = answers_from(load(root / "measure" / "M000" / "none"))
    c0 = (root / "C_000.md").read_text().strip()
    rows = []
    for t in range(state["completed_rounds"] + 1):
        ct = (root / f"C_{t:03d}.md").read_text().strip()
        row = {"round": t, "checkpoint": f"M{t:03d}", "words": len(ct.split()), "principles": len(parse_principles(ct)),
               "change_from_c0": change_ratio(c0, ct)}
        own, _ = answers_from(load(root / "measure" / f"M{t:03d}" / "none"))
        if own and t > 0:
            r = ea.permutation_w1(own, prior, n_perm=300); row.update({"trained_w1": r["observed_w1"], "trained_excess": r["excess"], "trained_p": r["p_value"]})
        post_dpo, _ = answers_from(load(root / "measure" / f"M{t:03d}_dpo" / "none"))
        if post_dpo and t > 0:
            r = ea.permutation_w1(post_dpo, prior, n_perm=300); row.update({"post_dpo_excess": r["excess"], "post_dpo_p": r["p_value"]})
        gap_path = root / f"round_{t:03d}" / "preferences.gap.json"
        if gap_path.exists():
            g = json.loads(gap_path.read_text()); row.update({"pref_pairs": g["pairs"], "pref_length_ratio": g["length_ratio"],
                                                              "pref_chosen_win_rate": g.get("chosen_win_rate"), "dpo_skipped": (root / f"round_{t:03d}" / "dpo_skipped.json").exists()})
        rv = root / f"round_{t:03d}" / "review.json"
        if rv.exists():
            row["edit_kind"] = json.loads(rv.read_text()).get("kind")
        prompted, _ = answers_from(load(root / "measure" / "M000" / f"prompted_C{t:03d}"))
        if prompted:
            r = ea.permutation_w1(prompted, prior, n_perm=300); row.update({"prompted_w1": r["observed_w1"], "prompted_excess": r["excess"], "prompted_p": r["p_value"]})
            if own and t > 0:
                r2 = ea.permutation_w1(own, prompted, n_perm=300); row.update({"trained_vs_prompted_excess": r2["excess"], "trained_vs_prompted_p": r2["p_value"]})
        rows.append(row)
    steps = pd.DataFrame(rows)
    res = root / "results"; res.mkdir(exist_ok=True)
    if eigen and state["completed_rounds"] > 0:
        from valuedrift.eigen import build_evaluations, eigen_judge_task, eigen_responses_task, fit_scores, load_scenarios, make_pairs, responses_from_log
        serve_cfg = config["serve"]; scenarios = load_scenarios(limit=bench.get("scenarios_limit", 20)); responses = {}
        for t in range(state["completed_rounds"] + 1):
            ckpt = config["model"] if t == 0 else str(root / f"round_{t:03d}" / "final")
            with ServedModel(ckpt, serve_cfg, f"M{t:03d}", log_dir=root / "serve-logs") as served:
                log = _inspect_eval(eigen_responses_task("none", entity=f"M{t:03d}", scenarios_limit=bench.get("scenarios_limit", 20), reasoning=bench.get("reasoning", "default")),
                                    served.model, res / "eigen" / "responses" / f"M{t:03d}", bench)
                responses[f"M{t:03d}"] = responses_from_log(log)
        pairs = make_pairs(responses, ["M000"], scenarios, random_pairs_per_scenario=bench.get("random_pairs", 2))
        pf = res / "eigen_pairs.jsonl"; pf.write_text("\n".join(json.dumps(p) for p in pairs) + "\n")
        scores = {}
        for ref_t in sorted({0, state["completed_rounds"]}):
            ref = str(root / f"C_{ref_t:03d}.md")
            jlog = _inspect_eval(eigen_judge_task(str(pf), ref), bench["judge"], res / "eigen" / f"judgments_C{ref_t:03d}", bench)
            names = build_evaluations(jlog, "judge", res / f"eigen_evaluations_C{ref_t:03d}.jsonl")
            scores[f"C{ref_t:03d}"] = fit_scores(jlog, names, "judge")
        (res / "eigen_scores.json").write_text(json.dumps(scores, indent=1))
        for ref, sc in scores.items():
            steps[f"eigen_vs_{ref}"] = [sc.get(f"M{t:03d}") for t in steps["round"]]
    steps.to_csv(res / "steps.csv", index=False)
    return steps
