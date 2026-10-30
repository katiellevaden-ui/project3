"""Offline end-to-end tests of every stage on the scripted model, plus the analysis on the
logs those stages produce."""
import json
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval

import scripted_model
from valuedrift import analysis, registry
from valuedrift.chain import chain_task, export_versions
from valuedrift.eigen import (build_evaluations, comparisons_from_log, eigen_judge_task, eigen_responses_task,
                              fit_scores, load_scenarios, make_pairs, responses_from_log)
from valuedrift.elicit import elicit_task
from valuedrift.valuebench import load_items, parse_rating, split_items, valuebench_task

M, J = "scripted/model", "scripted/judge"
C0 = str(registry.CONSTITUTIONS_DIR / "c0_general_assistant.md")


def _run(task, tmp_path, model=M, roles=None, **kw):
    [log] = inspect_eval(task, model=model, model_roles=roles or {}, log_dir=str(tmp_path), display="none", **kw)
    assert log.status == "success", log.error
    return log


def label(**kw):
    return json.dumps({"model": "m1", "constitution": "c0_general_assistant", "chain": None, "round": None, "repeat": 0, **kw})


# --- registry and data ---

def test_registry_loads_models_and_constitutions():
    models = registry.load_models()
    assert len(models) >= 20 and all(m.inspect_name.startswith("openrouter/") for m in models)
    assert registry.load_models(nicks=["qwen3.5-9b"])[0].model_args == {}
    pinned = registry.load_models(nicks=["deepseek-v4-pro"])[0]
    assert pinned.model_args["provider"]["order"] == ["novita"]
    cs = registry.load_constitutions()
    names = [c.name for c in cs]
    assert "c0_general_assistant" in names and "prose_general_assistant" in names and names[-1] == "none"
    assert cs[-1].is_prior


def test_valuebench_items_split_and_parser():
    items = load_items()
    a, b = split_items(items)
    assert len(a) + len(b) == len(items) and not {i.id for i in a} & {i.id for i in b}
    assert parse_rating('{"explanation": "x", "rating": 9}') == 9 and parse_rating("rating: 11") is None


# --- stages ---

def test_elicit_writes_a_headed_constitution(tmp_path):
    log = _run(elicit_task("test-model", out_dir=str(tmp_path / "elicited")), tmp_path / "log")
    text = (tmp_path / "elicited" / "test-model.md").read_text()
    assert "### P1." in text and log.samples[0].scores["elicited_shape"].value["headed"] == 1.0


def test_valuebench_task_runs_and_orients(tmp_path):
    log = _run(valuebench_task(constitution=C0, questionnaires="LTO", label=label()), tmp_path, roles={"judge": J}, epochs=2)
    assert len(log.samples) == 16                               # 8 LTO items x 2 epochs
    vals = [s.scores["valuebench_rating"].value for s in log.samples]
    assert any(v["unparsed"] == 0 for v in vals)
    for s in log.samples:
        v = s.scores["valuebench_rating"].value
        if v["unparsed"] == 0:
            r, o, agr = v["rating"], v["oriented"], s.metadata["agreement"]
            assert o == (10 - r if agr == -1 else r)
    prior = _run(valuebench_task(constitution="none", questionnaires="LTO", label=label(constitution="none")),
                 tmp_path / "prior", roles={"judge": J})
    sys_msgs = [c["system"] for c in scripted_model.CALLS if c["user"].endswith("50 words.")]
    assert any(s == "" for s in sys_msgs) and any("### P1." in s for s in sys_msgs)


def test_chain_task_and_export(tmp_path):
    log = _run(chain_task(constitution=C0, rounds=3, label=label()), tmp_path / "log", epochs=2)
    idx = export_versions(log, tmp_path / "versions", "m1", "c0_general_assistant")
    assert len(idx) == 2 * 4 and all(Path(v["path"]).exists() for v in idx)
    assert all(v["round"] in range(4) for v in idx)
    v = log.samples[0].scores["chain_summary"].value
    assert v["rounds"] == 3 and v["errored"] == 0
    with pytest.raises(ValueError):
        chain_task(constitution="none")


def test_chain_task_accepts_a_custom_invitation(tmp_path):
    from valuedrift.chain import edit_messages
    sys_default, user_default = edit_messages("# C\n\n### P1. A\nbody", None)
    sys_custom, user_custom = edit_messages("# C\n\n### P1. A\nbody", "Please take a look at the constitution above.")
    assert sys_custom == sys_default and user_custom.startswith("Please take a look") and user_custom.endswith(user_default[-60:])
    log = _run(chain_task(constitution=C0, rounds=2, label=label(), invitation="Have a look at the constitution above.", invitation_id="t1"),
               tmp_path / "log")
    assert log.eval.metadata["invitation_id"] == "t1" and log.samples[0].scores["chain_summary"].value["rounds"] == 2


def test_eigen_pipeline_end_to_end(tmp_path):
    scenarios = load_scenarios(limit=4)
    responses = {}
    for ent, const in (("m1|none|base", "none"), ("m1|c0|base", C0), ("m1|c0|c1r3", C0)):
        log = _run(eigen_responses_task(constitution=const, entity=ent, scenarios_limit=4), tmp_path / ent.replace("|", "_"))
        responses[ent] = responses_from_log(log)
        assert len(responses[ent]) == 4
    pairs = make_pairs(responses, anchors=["m1|none|base", "m1|c0|base"], scenarios=scenarios, random_pairs_per_scenario=0)
    assert len(pairs) == 4 * 3 * 2                               # 3 unordered pairs x both orders x 4 scenarios
    pf = tmp_path / "pairs.jsonl"; pf.write_text("\n".join(json.dumps(p) for p in pairs))
    jlog = _run(eigen_judge_task(str(pf), C0), tmp_path / "judge", model=J, roles={"judge": J})
    names = build_evaluations(jlog, "judge", tmp_path / "evals.jsonl")
    assert set(names) == set(responses) | {"judge"}
    comps = comparisons_from_log(jlog, names, "judge")
    assert comps and all(len(c) == 6 and c[0] == 0 and c[5] in (0, 1, 2) for c in comps)
    scores = fit_scores(jlog, names, "judge", max_epochs=20)
    assert set(scores) == set(names) and all(s > 0 for s in scores.values())


def test_contradictory_orderings_become_ties():
    from inspect_ai.log import EvalLog
    class S:  # minimal stand-ins for samples
        def __init__(self, a, b, choice, sc=0):
            self.metadata = {"eval1_name": a, "eval2_name": b, "scenario_index": sc}
            self.scores = {"choice_record": type("x", (), {"metadata": {"choice": choice}})()}
    log = type("L", (), {"samples": [S("x", "y", 1), S("y", "x", 1)]})()     # judge preferred first both times
    comps = comparisons_from_log(log, ["judge", "x", "y"], "judge")
    assert all(c[5] == 0 for c in comps)
    log = type("L", (), {"samples": [S("x", "y", 1), S("y", "x", 2)]})()     # consistent: x wins both
    comps = comparisons_from_log(log, ["judge", "x", "y"], "judge")
    assert sorted(c[5] for c in comps) == [1, 2]


# --- analysis on real logs from the stages above ---

def test_analysis_on_stage_logs(tmp_path):
    logs = tmp_path / "vb"
    for const_name, const in (("none", "none"), ("c0_general_assistant", C0)):
        for repeat in (0, 1):
            _run(valuebench_task(constitution=const, questionnaires="LTO",
                                 label=label(constitution=const_name, repeat=repeat)),
                 logs / const_name / f"base{repeat}", roles={"judge": J}, epochs=2)
    for rnd in (1, 2):
        _run(valuebench_task(constitution=C0, questionnaires="LTO", label=label(chain=1, round=rnd)),
             logs / "c0" / f"r{rnd}", roles={"judge": J}, epochs=2)
    df = analysis.load_valuebench(logs)
    assert set(df.columns) >= {"entity", "model", "constitution", "round", "oriented", "value"}
    assert df.entity.nunique() == 6
    prof = analysis.dimension_profiles(df)
    assert set(prof.columns) == {"Planning", "Tradition"}
    drift = analysis.drift_table(df)
    assert len(drift) == 2 and drift["w1_from_round0"].notna().all() and drift["w1_from_prior"].notna().all()
    nf = analysis.noise_floor(df)
    assert len(nf) == 2
    D, pos = analysis.plot_trajectories(df, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").exists() and D.shape == (6, 6) and len(pos) == 6
    analysis.plot_eigen_scores({"a": 0.2, "b": 0.3}, tmp_path / "eig.png")
    assert (tmp_path / "eig.png").exists()
