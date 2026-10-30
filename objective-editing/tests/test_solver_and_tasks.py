"""End-to-end runs of the real task on the scripted model: free and offline."""
import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelAPI, modelapi

import scripted_model
from objective_editing import prompts
from objective_editing.data import build_conditions, load_scenarios
from objective_editing.solver import FIXED_POINT_STREAK, LoopConfig
from objective_editing.store import RoundRecord
from objective_editing.tasks import recursive_edit_task

M = "scripted/editor"


def _run(tmp_path, task, model=M, epochs=1, **kw):
    [log] = inspect_eval(task, model=model, log_dir=str(tmp_path), display="none", epochs=epochs, **kw)
    assert log.status == "success", log.error
    return log


class _Logged:
    """The chain state as written to the log. (Do not rebuild a ChainStore outside an
    eval: StoreModel objects bind to a live store and would overwrite each other.)"""
    def __init__(self, sample):
        st = sample.store
        self.c0 = st["ChainStore:c0"]
        self.versions = st["ChainStore:versions"]
        self.rounds = [RoundRecord.model_validate(r) for r in st["ChainStore:rounds"]]
        self.stop_reason = st.get("ChainStore:stop_reason")
        self.error = st.get("ChainStore:error")
        self.prompt_fingerprint = st.get("ChainStore:prompt_fingerprint")


def _store(sample):
    return _Logged(sample)


def test_task_builds_49_conditions_and_records_its_config():
    t = recursive_edit_task()
    assert len(t.dataset) == 49 and t.metadata["prompt_fingerprint"] == prompts.fingerprint()
    assert len(recursive_edit_task(tiers="aligned,mild", neutral=False, reflective=False).dataset) == 24
    assert len(recursive_edit_task(judges=True).scorer) == 3 and len(t.scorer) == 1
    with pytest.raises(ValueError):
        LoopConfig(rounds=0)


def test_chains_keep_the_store_invariants(tmp_path):
    scripted_model.CALLS.clear()
    log = _run(tmp_path, recursive_edit_task(rounds=4, max_tokens=777, neutral=False, reflective=False), limit=12)
    kinds_seen = set()
    for sample in log.samples:
        s = _store(sample)
        assert s.versions[0] == s.c0 == s.c0.strip()
        assert len(s.versions) == len(s.rounds) + 1 and 1 <= len(s.rounds) <= 4
        for r, prev, new in zip(s.rounds, s.versions, s.versions[1:]):
            kinds_seen.add(r.kind)
            assert r.changed == (new != prev)
            assert (r.kind == "edit") == r.changed                 # only edits change the text
            assert new == new.strip() and r.reply
        assert s.stop_reason in ("fixed_point", "max_rounds")
        if s.stop_reason == "fixed_point":
            assert [r.kind for r in s.rounds[-FIXED_POINT_STREAK:]] == ["unchanged"] * FIXED_POINT_STREAK
        else:
            assert len(s.rounds) == 4
        assert sample.scores["chain_scorer"].value["rounds"] == len(s.rounds)
    assert {"edit", "unchanged"} <= kinds_seen
    assert all(c["max_tokens"] == 777 for c in scripted_model.CALLS)


def test_each_round_sees_the_previous_rounds_constitution(tmp_path):
    scripted_model.CALLS.clear()
    log = _run(tmp_path, recursive_edit_task(rounds=3, neutral=False, reflective=False), limit=6)
    for sample in log.samples:
        s = _store(sample)
        cond = sample.metadata["condition_id"]
        calls = [c for c in scripted_model.CALLS if sample.metadata["objective"] in c["system"]]
        assert len(calls) == len(s.rounds)
        for t, call in enumerate(calls):                          # round t+1 is built on versions[t]
            assert f"<constitution>\n{s.versions[t]}\n</constitution>" in call["system"]
            assert call["user"] == calls[0]["user"]               # identical user message every round
            assert "misaligned" not in call["user"] and "conflicts" not in call["system"]


def test_reflective_and_neutral_controls_render(tmp_path):
    scripted_model.CALLS.clear()
    task = recursive_edit_task(rounds=1, tiers="aligned", neutral=True, reflective=True)
    log = _run(tmp_path, task, limit=None)
    ids = {s.id for s in log.samples}
    assert "reflective" in ids and "supplement_store__neutral" in ids
    reflective_calls = [c for c in scripted_model.CALLS if "the assistant you would want to be" in c["user"]]
    assert reflective_calls and all(prompts.DEPLOYMENT_HEADER not in c["system"] for c in reflective_calls)
    neutral_calls = [c for c in scripted_model.CALLS if prompts.NEUTRAL_OBJECTIVE in c["system"]]
    assert len(neutral_calls) == 12


@modelapi(name="failing")
def failing():
    class Failing(ModelAPI):
        async def generate(self, input, tools, tool_choice, config):
            raise RuntimeError("provider down")
    return Failing


def test_model_errors_are_recorded_not_fatal(tmp_path):
    log = _run(tmp_path, recursive_edit_task(rounds=2, neutral=False, reflective=False),
               model="failing/model", limit=1, max_retries=0)
    s = _store(log.samples[0])
    assert s.stop_reason == "error" and "provider down" in s.error and s.versions == [s.c0]
    assert log.samples[0].scores["chain_scorer"].value["errored"] == 1.0


def test_epochs_give_independent_chains(tmp_path):
    log = _run(tmp_path, recursive_edit_task(rounds=2, tiers="mild", neutral=False, reflective=False),
               epochs=2, limit=2)
    assert len(log.samples) == 4 and {s.epoch for s in log.samples} == {1, 2}

def test_framing_reaches_the_model_and_the_logs(tmp_path):
    scripted_model.CALLS.clear()
    log = _run(tmp_path, recursive_edit_task(rounds=1, framing="ownership", tiers="mild",
                                             neutral=False, reflective=False), limit=2)
    assert all(prompts.OWNERSHIP_SENTENCE in c["user"] for c in scripted_model.CALLS)
    assert log.eval.metadata["framing"] == "ownership"
    assert all(s.metadata["framing"] == "ownership" for s in log.samples)