"""End-to-end runs of the selection-pressure task on the scripted model."""
import math

import pytest
from inspect_ai import eval as inspect_eval

import scripted_model
from objective_editing import prompts
from objective_editing.selection import Candidate, SelectionConfig, SelectionRound, select
from objective_editing.tasks import selection_task
import random

M, J = "scripted/editor", "scripted/judge"


def _run(tmp_path, task, **kw):
    [log] = inspect_eval(task, model=M, model_roles={"judge": J}, log_dir=str(tmp_path), display="none", **kw)
    assert log.status == "success", log.error
    return log


def _store(sample):
    st = sample.store
    return {"c0": st["SelectionStore:c0"], "versions": st["SelectionStore:versions"],
            "rounds": [SelectionRound.model_validate(r) for r in st["SelectionStore:rounds"]],
            "baseline": st["SelectionStore:baseline_heldout"], "stop": st.get("SelectionStore:stop_reason"),
            "error": st.get("SelectionStore:error")}


def test_config_guards():
    with pytest.raises(ValueError): SelectionConfig(candidates=0)
    with pytest.raises(ValueError): SelectionConfig(selection="best")
    with pytest.raises(ValueError): SelectionConfig(temperature=0)
    with pytest.raises(ValueError): SelectionConfig(embodiment="both")


def test_select_rules():
    cands = [Candidate(index=i, kind="edit", constitution=f"c{i}", fitness=f) for i, f in enumerate([3.0, 9.0, 9.0])]
    rng = random.Random(0)
    assert select(cands, SelectionConfig(selection="max"), rng) in (1, 2)
    picks = {select(cands, SelectionConfig(selection="random"), random.Random(k)) for k in range(50)}
    assert picks == {0, 1, 2}
    soft = [select(cands, SelectionConfig(selection="softmax", temperature=0.1), random.Random(k)) for k in range(50)]
    assert 0 not in soft                                            # low temperature ~ max
    cands[1] = Candidate(index=1, kind="parse_failure", constitution="c1", fitness=None)
    assert select(cands, SelectionConfig(selection="max"), rng) == 2   # None fitness never wins


def test_selection_chain_invariants(tmp_path):
    scripted_model.CALLS.clear()
    log = _run(tmp_path, selection_task(rounds=2, candidates=3, tiers="mild"), limit=3)
    for sample in log.samples:
        st = _store(sample)
        assert st["stop"] == "max_rounds", st["error"]
        assert len(st["versions"]) == len(st["rounds"]) + 1 == 3
        assert set(st["baseline"]) >= {"objective", "adherence_conflicts", "adherence_collateral", "responses"}
        for t, r in enumerate(st["rounds"], 1):
            assert len(r.candidates) == 3 and 0 <= r.selected < 3
            sel = r.candidates[r.selected]
            assert st["versions"][t] == sel.constitution and r.changed == (sel.constitution != st["versions"][t - 1])
            fits = [c.fitness for c in r.candidates if c.fitness is not None]
            if fits and sel.fitness is not None:
                assert sel.fitness == max(fits)                        # selection="max"
            assert all(len(c.fitness_scores) == 5 for c in r.candidates)   # 5 fitness requests
            assert len(r.heldout["responses"]) == 3                        # 3 held-out requests
            for c in r.candidates:
                if c.kind != "edit":
                    assert c.constitution == st["versions"][t - 1]
        v = sample.scores["selection_scorer"].value
        assert v["rounds"] == 2 and not math.isnan(v["final_objective"]) and v["errored"] == 0
        series = sample.scores["selection_scorer"].metadata["series"]
        assert len(series) == 2 and "heldout_adherence_collateral" in series[0]
    # judges never see the constitution
    judge_calls = [c for c in scripted_model.CALLS if c["model"] == "judge"]
    assert judge_calls and all("<constitution>" not in c["user"] and "### P1." not in c["user"] for c in judge_calls)
    # actors see the constitution in the system prompt; editors got K parallel proposals per round
    actor_calls = [c for c in scripted_model.CALLS if c["model"] == "editor" and "<decision>" not in c["user"]]
    editor_calls = [c for c in scripted_model.CALLS if c["model"] == "editor" and "<decision>" in c["user"]]
    assert actor_calls and all("<constitution>" in c["system"] for c in actor_calls)
    assert len(editor_calls) == 3 * 2 * 3                              # samples x rounds x candidates


def test_random_and_softmax_selection_run(tmp_path):
    for sel in ("random", "softmax"):
        log = _run(tmp_path / sel, selection_task(rounds=1, candidates=2, selection=sel, tiers="mild"), limit=1)
        assert log.samples[0].scores["selection_scorer"].value["rounds"] == 1


def test_user_embodiment_in_selection(tmp_path):
    scripted_model.CALLS.clear()
    _run(tmp_path, selection_task(rounds=1, candidates=2, tiers="mild", embodiment="user"), limit=1)
    model_calls = [c for c in scripted_model.CALLS if c["model"] == "editor"]
    assert model_calls and all(c["system"] == "" and "<constitution>" in c["user"] for c in model_calls)


def test_collateral_excludes_conflicts_and_validates():
    with pytest.raises(ValueError, match="collateral principles"):
        selection_task(collateral="P2,P99")
    assert selection_task(collateral=["P2", "P4"]).metadata["collateral"] == ["P2", "P4"]
