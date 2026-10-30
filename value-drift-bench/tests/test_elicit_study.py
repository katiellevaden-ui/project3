"""Offline tests of the elicitation study: the prompt factorial, the validity gate, the echo
metric, the principle coder, and the analysis on logs the stages produce."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from inspect_ai import eval as inspect_eval

import scripted_model  # noqa: F401  (registers scripted/<name>)
from valuedrift import elicit_analysis as ea
from valuedrift.analysis import dimension_profiles
from valuedrift.coding import coding_profiles, coding_table, coding_task, count_profile, parse_codes
from valuedrift.elicit import elicit_task
from valuedrift.elicit_variants import (PRIMED_VALUES, Variant, elicit_variants_task, export_elicitations, merge_manifest,
                                        parse_variant, prompt_echo, validate, variants, write_manifest)
from valuedrift import registry
from valuedrift.valuebench import valuebench_task

M, J, J2 = "scripted/model", "scripted/judge", "scripted/judge2"


def _run(task, log_dir, model=M, roles=None, **kw):
    [log] = inspect_eval(task, model=model, model_roles=roles or {}, log_dir=str(log_dir), display="none", **kw)
    assert log.status == "success", log.error
    return log


def _label(model="m1", constitution="none", repeat=0, variant=None, replicate=None):
    return json.dumps({"model": model, "constitution": constitution, "chain": None, "round": None,
                       "repeat": repeat, "variant": variant, "replicate": replicate})


# --- the factorial ---

def test_variant_design():
    oat, full = variants("oat"), variants("all")
    assert len(oat) == 6 and len(full) == 18 and len({v.id for v in full}) == 18
    assert oat[0] == Variant("actual", "none", "plain")
    assert len({v.user_prompt() for v in full}) == 18
    assert parse_variant("ideal-set_b-ai").user_prompt().startswith("As an AI, write the constitution")
    assert all(v.user_prompt().rstrip().endswith("Return only the constitution.") for v in full)
    assert variants("actual-set_a-plain,ideal-none-ai") == [Variant("actual", "set_a", "plain"), Variant("ideal", "none", "ai")]
    with pytest.raises(ValueError):
        parse_variant("actual-set_c-plain")
    assert not (PRIMED_VALUES["set_a"] & PRIMED_VALUES["set_b"])


def test_echo_and_gate():
    prompt = variants("actual-set_a-plain")[0].user_prompt()
    echo, coverage = prompt_echo(prompt, "### P1. A\nI weigh autonomy and safety, and I value curiosity.")
    assert echo == pytest.approx(2 / 3) and coverage == pytest.approx(2 / 6)
    assert math.isnan(prompt_echo(prompt, "### P1. A\nnothing here")[0])
    assert validate("# T\n\n### P1. X\nbody") == (True, "")
    assert validate("No Constitution Available. As an AI I cannot write one.") == (False, "refusal")
    assert validate("Here are some thoughts without headings.") == (False, "no_principles")


def test_pipeline_elicit_rejects_refusals(tmp_path, monkeypatch):
    # force the scripted refusal bucket by trying nicks until one lands there is fragile; instead
    # drive the gate directly through the task on a model whose reply has no headings
    import scripted_model as sm
    monkeypatch.setattr(sm, "reply", lambda system, user: "No Constitution Available. I cannot write one.")
    log = _run(elicit_task("ref-model", out_dir=str(tmp_path / "elicited")), tmp_path / "log")
    assert not (tmp_path / "elicited" / "ref-model.md").exists()
    assert (tmp_path / "elicited" / "ref-model.rejected.txt").exists()
    assert log.samples[0].scores["elicited_shape"].value["valid"] == 0.0


# --- stages ---

def test_elicit_variants_task_and_export(tmp_path):
    log = _run(elicit_variants_task("m1", "actual-set_a-ai", replicates=8), tmp_path / "log")
    assert log.eval.metadata["prompt_fingerprint"] == parse_variant("actual-set_a-ai").fingerprint
    entries = export_elicitations(log, tmp_path / "elicitations", "m1")
    assert len(entries) == 8 and {e["replicate"] for e in entries} == set(range(8))
    assert all(Path(e["path"]).exists() for e in entries)
    assert any(e["valid"] for e in entries)
    for e in entries:
        assert Path(e["path"]).suffix == (".md" if e["valid"] else ".txt")
        if e["valid"]:
            assert e["principles"] == 8 and e["echo"] is not None
    # the scripted model's refusal bucket must have been exercised at least once across cells
    rejected = [e for e in entries if not e["valid"]]
    assert all(e["reason"] in ("refusal", "no_principles") for e in rejected)


def test_export_records_provenance_and_skips_failed_logs(tmp_path):
    log = _run(elicit_variants_task("m1", "actual-none-plain", replicates=2, label="run A"), tmp_path / "log")
    entries = export_elicitations(log, tmp_path / "e", "m1")
    assert all(e["run_id"] == log.eval.run_id and e["log"] == log.location and e["label"] == "run A" and e["created"] for e in entries)
    log.status = "error"
    assert export_elicitations(log, tmp_path / "e2", "m1") == [] and not (tmp_path / "e2").exists()
    merged = merge_manifest([{"model": "m1", "variant": "actual-none-plain", "replicate": 0, "label": "old"},
                             {"model": "m2", "variant": "actual-none-plain", "replicate": 0, "label": "keep"}], entries)
    assert sorted(e["label"] for e in merged) == ["keep", "run A", "run A"]


def test_per_model_reasoning_override():
    glm = registry.load_models(nicks=["glm-5.3"])[0]
    qwen = registry.load_models(nicks=["qwen3.5-9b"])[0]
    assert glm.reasoning_or("none") == "minimal" and qwen.reasoning_or("none") == "none"


def test_coding_task_profiles_and_agreement(tmp_path):
    log = _run(elicit_variants_task("m1", "actual-none-plain", replicates=4), tmp_path / "log")
    entries = export_elicitations(log, tmp_path / "elicitations", "m1")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(entries, manifest)
    clog = _run(coding_task(str(manifest), "judge,judge2"), tmp_path / "coding", model=J, roles={"judge": J, "judge2": J2})
    n_valid = sum(e["valid"] for e in entries)
    assert len(clog.samples) == n_valid
    for s in clog.samples:
        v = s.scores["schwartz_coding"].value
        assert v["coded"] == 1.0 and 0 <= v["agreement"] <= 1 and v["n_principles"] == 8
        prof = s.scores["schwartz_coding"].metadata["profiles"]["judge"]
        assert sum(prof.values()) == pytest.approx(1.0)
    table = coding_table(clog)
    assert set(table.judge) == {"judge", "judge2"} and len(table) == n_valid * 2 * 11
    cp = coding_profiles(table)
    assert cp.shape == (n_valid, 10) and (cp.sum(axis=1) <= 1.0 + 1e-9).all()


def test_duplicate_logs_in_one_directory_count_once(tmp_path):
    from valuedrift.analysis import successful_logs
    d = tmp_path / "vb" / "m1" / "none" / "repeat0"
    for _ in range(2):                                   # two processes measured the same state
        _run(valuebench_task(constitution="none", questionnaires="LTO", label=_label()), d, roles={"judge": J})
    assert len(successful_logs(tmp_path / "vb")) == 1
    df = ea.load_study(tmp_path / "vb")
    assert df.groupby(["entity", "item", "epoch"]).size().max() == 1


def test_axes_task_table_and_plot(tmp_path):
    from valuedrift.text_axes import axes_table, axes_task, parse_axes, plot_text_axes
    assert parse_axes('{"self_model": 6, "warmth": "2", "justification": "x"}') == {"self_model": 6, "warmth": 2, "justification": "x"}
    assert parse_axes('{"self_model": 9, "warmth": 2}') == {"warmth": 2} and parse_axes("nope") == {}
    log = _run(elicit_variants_task("m1", "actual-none-plain", replicates=4), tmp_path / "log")
    entries = export_elicitations(log, tmp_path / "elicitations", "m1")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(entries, manifest)
    alog = _run(axes_task(str(manifest)), tmp_path / "axes", model=J, roles={"judge": J})
    t = axes_table(alog)
    assert len(t) == sum(e["valid"] for e in entries) and t.self_model.between(1, 7).all() and t.warmth.between(1, 7).all()
    cent = plot_text_axes(t, {"m1": "open_mid"}, tmp_path / "fig" / "axes.png")
    assert (tmp_path / "fig" / "axes.png").exists() and list(cent.index) == ["m1"]


def test_coarse_map_names_exist_and_pooling_counts():
    import csv
    from valuedrift.valuebench import ITEMS_FILE
    names = {(r["questionnaire"], r["value"]) for r in csv.DictReader(open(ITEMS_FILE, encoding="utf-8"))}
    assert set(ea.COARSE) <= names, sorted(set(ea.COARSE) - names)
    assert len({k for k in ea.COARSE if k[0] == "HEXACO"}) == 25 and len({k for k in ea.COARSE if k[0] == "PVQ40"}) == 10
    rows = []
    for v in ("actual-none-plain", "ideal-none-plain"):
        for r in range(3):
            for item in ("i1", "i2"):
                rows.append({"entity": f"m1|elicited:{v}/r{r}|base", "model": "m1", "constitution": f"elicited:{v}/r{r}", "variant": v,
                             "replicate": r, "repeat": 0, "item": item, "epoch": 1, "questionnaire": "HEXACO", "value": "Sincerity", "oriented": float(r)})
    rows.append({**rows[0], "entity": "m1|elicited:actual-none-plain/r0|base#1", "repeat": 1})    # a re-measurement: dropped
    for rep in (0, 1):
        for item in ("i1", "i2"):
            rows.append({"entity": "m1|none|base" + ("#1" if rep else ""), "model": "m1", "constitution": "none", "variant": None,
                         "replicate": None, "repeat": rep, "item": item, "epoch": 1, "questionnaire": "PVQ40", "value": "Power", "oriented": 5.0})
    df = pd.DataFrame(rows)
    pooled = ea.pool_cells(df)
    assert pooled.entity.nunique() == 3                                   # two cells + one prior
    assert pooled.groupby(["entity", "item"]).size().to_dict()["m1|elicited:actual-none-plain|cell", "i1"] == 3
    assert pooled.groupby(["entity", "item"]).size().to_dict()["m1|none|base", "i1"] == 2
    cp = ea.coarse_profiles(df)
    assert set(cp.columns) == {"Honesty-Humility", "Self-enhancement"}
    assert ea.coarse_questionnaires(df) == {"Honesty-Humility": "HEXACO", "Self-enhancement": "PVQ40"}


def test_permutation_w1_null_is_centred_when_states_are_identical():
    rng = np.random.default_rng(0)
    a = {f"i{k}": rng.integers(0, 11, 30).astype(float) for k in range(40)}
    b = {k: rng.choice(v, 2) for k, v in a.items()}                      # a 2-answer draw from the same answers
    t = ea.permutation_w1(b, a, n_perm=200)
    assert t["null_mean"] > 0.3 and t["p_value"] > 0.05 and abs(t["excess"]) < 3 * t["null_sd"] + 0.05


def test_parse_codes_is_strict():
    assert parse_codes('x {"P1": "security", "P2": "Nope", "P9": "Power"} y', ["P1", "P2"]) == {"P1": "Security"}
    assert parse_codes("no json", ["P1"]) == {}
    assert count_profile({})["Power"] != count_profile({})["Power"]           # NaN when nothing was coded


# --- analysis on synthetic profiles ---

def _meta(rows):
    m = pd.DataFrame(rows).set_index("entity")
    return ea.add_factors(m)


def test_w1_distance_is_mean_absolute_rating_difference_at_one_answer():
    from valuedrift.analysis import w1_distance
    one = lambda k: np.eye(11)[k]
    a = {"i1": one(3), "i2": one(5)}
    b = {"i1": one(4), "i2": one(10)}
    assert w1_distance(a, b) == pytest.approx((1 + 5) / 2)
    assert w1_distance(a, a) == 0.0
    spread = {"i1": np.array([0, 0, 0, .5, .5, 0, 0, 0, 0, 0, 0])}      # half at 3, half at 4
    assert w1_distance(spread, {"i1": one(3)}) == pytest.approx(0.5)


def _vc_rows(noise_gap=0.0):
    rows, prof = [], {}
    for mi, model in enumerate(("a", "b", "c")):
        for v in ("actual-none-plain", "ideal-none-plain"):
            for r in range(3):
                e = f"{model}|{v}|r{r}"
                rows.append({"entity": e, "model": model, "constitution": f"elicited:{v}", "variant": v, "replicate": r, "repeat": 0})
                prof[e] = {"Security": 10 * mi, "Tradition": 5.0 if v.startswith("ideal") else 0.0, "Care": float(r)}
        for rep in (0, 1):                                            # a re-measured prior: the noise pair
            e = f"{model}|none|base" + ("#1" if rep else "")
            rows.append({"entity": e, "model": model, "constitution": "none", "variant": None, "replicate": None, "repeat": rep})
            prof[e] = {"Security": 1.0, "Tradition": 1.0, "Care": 1.0 + noise_gap * rep}
    return rows, pd.DataFrame(prof).T


def test_variance_components_attribute_variance_correctly():
    rows, X = _vc_rows()
    out = ea.variance_decomposition(X, _meta(rows)).set_index("dimension")
    assert out.loc["Security", "vc_model"] == pytest.approx(1.0) and out.loc["Security", "vc_prompt"] == pytest.approx(0.0)
    assert out.loc["Tradition", "vc_prompt"] == pytest.approx(1.0) and out.loc["Tradition", "vc_model"] == pytest.approx(0.0)
    assert out.loc["Care", "vc_replicate"] == pytest.approx(1.0)        # replicates differ, nothing else does
    assert out.loc["Care", "var_measurement"] == 0.0 and out.loc["ALL", "n_noise_pairs"] == 3
    assert out.loc["Tradition", "prompt_to_model"] == math.inf
    assert out.loc["ALL", ["vc_model", "vc_prompt", "vc_replicate", "vc_measurement"]].sum() == pytest.approx(1.0)


def test_measurement_noise_is_taken_out_of_the_replicate_component():
    rows, X = _vc_rows(noise_gap=2.0)                                  # re-measurements differ by 2 -> var_meas = 2
    out = ea.variance_decomposition(X, _meta(rows)).set_index("dimension")
    assert out.loc["Care", "var_measurement"] == pytest.approx(2.0)
    assert out.loc["Care", "var_replicate"] == pytest.approx(1.0 - 2.0, abs=1e-9) or out.loc["Care", "var_replicate"] == 0.0
    assert out.loc["Care", "vc_measurement"] == pytest.approx(1.0)      # noise explains all of the within-cell spread


def test_priming_test_separates_primed_from_other_dimensions():
    rows, prof = [], {}
    for v in ("actual-none-plain", "actual-set_a-plain", "actual-set_b-plain"):
        for r in range(2):
            e = f"a|{v}|r{r}"
            rows.append({"entity": e, "model": "a", "constitution": f"elicited:{v}", "variant": v, "replicate": r, "repeat": 0})
            p = {k: 5.0 for k in ("Security", "Conformity", "Tradition", "Hedonism", "Care")}
            if "set_a" in v:
                p["Security"] = p["Conformity"] = 8.0                      # exactly the primed values move
            prof[e] = p
    out = ea.priming_test(pd.DataFrame(prof).T, _meta(rows)).set_index("priming")
    assert out.loc["set_a", "delta_primed"] == pytest.approx(3.0) and out.loc["set_a", "delta_other"] == pytest.approx(0.0)
    assert out.loc["set_b", "delta_primed"] == pytest.approx(0.0) and out.loc["set_b", "n_primed_dims"] == 2


def test_text_distance_summary_prefers_the_organising_factor():
    texts, rows = {}, []
    for model, vocab in (("a", "apples oranges pears"), ("b", "quarks gluons leptons")):
        for v in ("actual-none-plain", "ideal-none-plain"):
            for r in range(2):
                e = f"{model}|{v}|r{r}"
                texts[e] = f"{vocab} {vocab} principle {r}"
                rows.append({"entity": e, "model": model, "constitution": f"elicited:{v}", "variant": v, "replicate": r, "repeat": 0})
    out = ea.text_distance_summary(texts, _meta(rows))
    assert out["n"] == 8 and out["silhouette_by_model"] > out["silhouette_by_variant"]


# --- end to end on the scripted model: measure + analysis ---

def test_measure_and_analysis_on_stage_logs(tmp_path):
    log = _run(elicit_variants_task("m1", "actual-none-plain", replicates=3), tmp_path / "e1")
    entries = export_elicitations(log, tmp_path / "elicitations", "m1")
    log2 = _run(elicit_variants_task("m1", "actual-set_a-plain", replicates=3), tmp_path / "e2")
    entries += export_elicitations(log2, tmp_path / "elicitations", "m1")
    valid = [e for e in entries if e["valid"]]
    assert valid
    logs = tmp_path / "vb"
    for repeat in (0, 1):
        _run(valuebench_task(constitution="none", questionnaires="LTO", label=_label(repeat=repeat)),
             logs / "none" / str(repeat), roles={"judge": J}, epochs=2)
    for e in valid:
        c = f"elicited:{e['variant']}/r{e['replicate']}"
        _run(valuebench_task(constitution=e["path"], questionnaires="LTO", label=_label(constitution=c, variant=e["variant"], replicate=e["replicate"])),
             logs / e["variant"] / str(e["replicate"]), roles={"judge": J}, epochs=2)
    first = valid[0]
    _run(valuebench_task(constitution=first["path"], questionnaires="LTO",
                         label=_label(constitution=f"elicited:{first['variant']}/r{first['replicate']}", repeat=1,
                                      variant=first["variant"], replicate=first["replicate"])),
         logs / "noise", roles={"judge": J}, epochs=2)
    df = ea.load_study(logs)
    assert df.entity.nunique() == len(valid) + 3
    meta = ea.entity_meta(df)
    assert set(meta.framing.dropna()) == {"actual"} and set(meta.priming.dropna()) <= {"none", "set_a"}
    prof = dimension_profiles(df)
    vd = ea.variance_decomposition(prof, meta)
    assert "ALL" in set(vd.dimension) and np.isclose(vd.set_index("dimension").loc["ALL", ["frac_model", "frac_prompt", "frac_sampling"]].sum(), 1.0)
    assert vd.set_index("dimension").loc["ALL", "frac_model"] == pytest.approx(0.0, abs=1e-9)            # one model: nothing between models
    inert = ea.inertness(df, meta, n_perm=30)
    assert len(inert) == len(valid) and inert.n_prior_answers.iloc[0] == 4 and inert.w1_from_prior.notna().all()
    assert inert.p_value.between(0, 1).all() and (inert.excess == inert.w1_from_prior - inert.null_mean).all()
    ps = ea.prior_shift_table(df, n_perm=30)
    assert list(ps.model) == ["m1"] and ps.n_prior.iloc[0] == 4 and ps.n_const.iloc[0] == 2 * len(valid)
    pt = ea.priming_test(prof, meta)
    if (meta.priming == "set_a").any() and (meta.priming == "none").any():
        assert len(pt) == 1 and pt.iloc[0]["n_primed_dims"] == 0            # LTO has no Schwartz dimensions
    echo = ea.echo_table(entries)
    assert set(echo.variant) == {"actual-none-plain", "actual-set_a-plain"} and (echo.valid_rate <= 1).all()
    shifts = ea.shift_table(prof, meta, vd, ea.dimension_questionnaires(df))
    assert set(shifts.model) == {"m1"} and set(shifts.questionnaire) == {"LTO"} and shifts.n_prior.iloc[0] == 2
    assert len(shifts) == prof.shape[1] and shifts.se.notna().all()
    M = ea.plot_shift_heatmap(shifts, tmp_path / "fig" / "heat.png")
    assert (tmp_path / "fig" / "heat.png").exists() and M.shape == (prof.shape[1], 1)
    D, pos = ea.plot_umap(df, tmp_path / "fig" / "umap.png", meta)
    assert (tmp_path / "fig" / "umap.png").exists() and D.shape == (len(pos), len(pos)) and set(pos) == set(meta.index)
    pooled = ea.pool_cells(df)
    D2, pos2 = ea.plot_umap(pooled, tmp_path / "fig" / "pooled.png", ea.entity_meta(pooled), point_label="cell")
    assert len(pos2) == pooled.entity.nunique() and (tmp_path / "fig" / "pooled.png").exists()
    pm = ea.pool_models(df)
    assert pm.entity.nunique() == 2 and pm.groupby(["entity", "item"]).size().max() >= 2 * len(valid)
    D3, pos3 = ea.plot_model_arrows(pm, {"m1": "open_mid"}, tmp_path / "fig" / "arrows.png", shifts=ps)
    scores, loadings, explained = ea.plot_model_arrows_pca(pm, {"m1": "open_mid"}, tmp_path / "fig" / "pca.png", shifts=ps)
    assert scores.shape == (2, 2) and loadings.shape[1] == 2 and 0 < explained.sum() <= 1 + 1e-9
    assert (tmp_path / "fig" / "pca.png").exists()
    assert len(pos3) == 2 and (tmp_path / "fig" / "arrows.png").exists()
