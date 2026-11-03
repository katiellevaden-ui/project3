"""Offline end-to-end run of the bench loop: scripted serving, fake trainer, one round."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "value-drift-bench" / "tests"))
import scripted_model  # noqa: E402  registers scripted/<name>
from recursive_oct.bench_loop import bench_report, run_bench_trajectory  # noqa: E402


class FakeTrainer:
    def __init__(self): self.calls = []
    def preferences(self, checkpoint, constitution_path, output):
        self.calls.append(("preferences", str(checkpoint), Path(constitution_path).read_text()[:40]))
        rows = [{"id": f"p{i}", "prompt": f"question {i}", "chosen": f"a careful answer with several words {i}", "rejected": f"a short answer {i}"} for i in range(6)]
        Path(output).write_text("".join(json.dumps(r) + "\n" for r in rows))
    def dpo(self, checkpoint, data, output):
        self.calls.append(("dpo", str(checkpoint))); Path(output).mkdir(parents=True, exist_ok=True); return str(output)
    def introspection(self, checkpoint, output, constitution_text):
        self.calls.append(("introspection", str(checkpoint), constitution_text[:40])); Path(output).write_text("{}\n")
    def sft(self, checkpoint, data, output):
        self.calls.append(("sft", str(checkpoint))); Path(output).mkdir(parents=True, exist_ok=True); return str(output)


@pytest.fixture
def config():
    cfg = json.loads((HERE.parent / "configs" / "bench-oct-smoke.json").read_text())
    cfg["bench"]["invitation_path"] = str(HERE.parent / cfg["bench"]["invitation_path"])
    return cfg


def test_two_rounds_end_to_end(tmp_path, config, monkeypatch):
    original = scripted_model.reply
    def always_edit(system, user):                                   # the scripted editor would otherwise flip a coin
        if system.startswith("You are acting as a judge"):          # and the judge must prefer the chosen ("careful") answer
            first = user.split("<first_response>", 1)[1].split("</first_response>", 1)[0]
            return f"<choice>{1 if 'careful' in first else 2}</choice>"
        if "<decision>" in user:
            current = (system if "<constitution>" in system else user).split("<constitution>\n", 1)[1].split("\n</constitution>", 1)[0]
            return f"<decision>REVISED</decision><constitution>{current}\n\n### P99. Added\nOne more.</constitution><justification>Added.</justification>"
        return original(system, user)
    monkeypatch.setattr(scripted_model, "reply", always_edit)
    trainer = FakeTrainer()
    state = run_bench_trajectory(tmp_path / "run", config, trainer)
    assert state["status"] == "ROUND_LIMIT" and state["completed_rounds"] == 2, state
    root = tmp_path / "run"
    assert (root / "C_000.md").exists() and "### P1." in (root / "C_000.md").read_text()       # elicited by M0
    assert (root / "C_001.md").read_text().strip() == (root / "C_000.md").read_text().strip()   # round 1 trains on the unedited C_0
    assert json.loads((root / "round_001" / "review.json").read_text())["kind"] == "initial"
    assert "P99" in (root / "C_002.md").read_text()                                             # round 2 edits C_1 -> C_2
    assert [c[0] for c in trainer.calls] == ["preferences", "dpo", "introspection", "sft"] * 2
    assert trainer.calls[0][1] == "M0" and trainer.calls[4][1].endswith("round_001/final")      # M_1 edits and trains next
    assert (root / "round_002" / "evidence.txt").read_text().startswith("Before deciding")
    assert not (root / "round_001" / "evidence.txt").exists()                                    # no edit in the initial round
    for n in (1, 2):
        assert (root / f"round_{n:03d}" / "elicited_after_training" / f"m{n:03d}.md").exists()   # safe_name lower-cases the tag
        gap = json.loads((root / f"round_{n:03d}" / "preferences.gap.json").read_text())
        assert gap["judged_pairs"] > 0 and 0 <= gap["chosen_win_rate"] <= 1 and gap["length_ratio"] > 1
        assert len(list((root / "measure" / f"M{n:03d}_dpo" / "none").glob("repeat*"))) == 2        # post-DPO measurement
    for d in ("M000/none", "M000/prompted_C000", "M001/none", "M000/prompted_C001", "M002/none", "M000/prompted_C002"):
        assert len(list((root / "measure" / d).glob("repeat*"))) == 2, d
    assert len(state["preference_gaps"]) == 2
    steps = bench_report(root, config, eigen=True)
    assert list(steps["round"]) == [0, 1, 2] and steps.loc[2, "trained_excess"] == steps.loc[2, "trained_excess"]
    assert "post_dpo_excess" in steps and "eigen_vs_C000" in steps and "eigen_vs_C002" in steps
    assert list(steps["edit_kind"].fillna("")) == ["", "initial", "edit"]
    assert run_bench_trajectory(root, config, FakeTrainer(), resume=True)["status"] == "ROUND_LIMIT"


def test_fixed_constitution_control_never_edits(tmp_path, config):
    cfg = dict(config); cfg["edit"] = False; cfg["train_initial"] = False; cfg["max_rounds"] = 2
    trainer = FakeTrainer()
    state = run_bench_trajectory(tmp_path / "run", cfg, trainer)
    assert state["status"] == "ROUND_LIMIT" and state["completed_rounds"] == 2
    root = tmp_path / "run"
    texts = {(root / f"C_{k:03d}.md").read_text().strip() for k in range(3)}
    assert len(texts) == 1 and all(r["kind"] == "fixed" for r in state["reviews"])
    assert [c[0] for c in trainer.calls] == ["preferences", "dpo", "introspection", "sft"] * 2


def test_unchanged_edit_stops_after_the_initial_round(tmp_path, config, monkeypatch):
    original = scripted_model.reply
    monkeypatch.setattr(scripted_model, "reply", lambda s, u: "<decision>UNCHANGED</decision><justification>Fine.</justification>" if "<decision>" in u else original(s, u))
    trainer = FakeTrainer()
    state = run_bench_trajectory(tmp_path / "run", config, trainer)
    assert state["status"] == "SELF_DECLARED_CONVERGENCE" and state["completed_rounds"] == 1
    assert [c[0] for c in trainer.calls] == ["preferences", "dpo", "introspection", "sft"]      # the initial C_0 round only


def test_collapsed_margin_skips_dpo_but_not_sft(tmp_path, config, monkeypatch):
    original = scripted_model.reply
    def tie_judge(system, user):                                     # every pair judged a tie -> win rate 0.5 < 0.6
        if system.startswith("You are acting as a judge"):
            return "Equivalent. <choice>0</choice>"
        if "<decision>" in user:
            current = (system if "<constitution>" in system else user).split("<constitution>\n", 1)[1].split("\n</constitution>", 1)[0]
            return f"<decision>REVISED</decision><constitution>{current}\n\n### P99. Added\nOne more.</constitution><justification>Added.</justification>"
        return original(system, user)
    monkeypatch.setattr(scripted_model, "reply", tie_judge)
    trainer = FakeTrainer()
    state = run_bench_trajectory(tmp_path / "run", config, trainer)
    assert state["status"] == "ROUND_LIMIT" and state["completed_rounds"] == 2
    assert [c[0] for c in trainer.calls] == ["preferences", "introspection", "sft"] * 2          # DPO skipped, SFT still ran
    assert [d["round"] for d in state["dpo_skipped"]] == [1, 2]
    assert (tmp_path / "run" / "round_001" / "dpo_skipped.json").exists()
    assert not (tmp_path / "run" / "measure" / "M001_dpo").exists()                               # nothing new to measure
    steps = bench_report(tmp_path / "run", config, eigen=False)
    assert list(steps["dpo_skipped"].fillna(False)) == [False, True, True]


def test_inference_only_arm_edits_without_training(tmp_path, config, monkeypatch):
    original = scripted_model.reply
    def always_edit(system, user):
        if "<decision>" in user:
            current = (system if "<constitution>" in system else user).split("<constitution>\n", 1)[1].split("\n</constitution>", 1)[0]
            return f"<decision>REVISED</decision><constitution>{current}\n\n### P99. Added\nOne more.</constitution><justification>Added.</justification>"
        return original(system, user)
    monkeypatch.setattr(scripted_model, "reply", always_edit)
    cfg = dict(config); cfg["train"] = False; cfg["max_rounds"] = 2
    trainer = FakeTrainer()
    state = run_bench_trajectory(tmp_path / "run", cfg, trainer)
    assert state["status"] == "ROUND_LIMIT" and state["completed_rounds"] == 2 and trainer.calls == []
    root = tmp_path / "run"
    assert (root / "C_002.md").read_text().count("P99") == 2                                      # two successive edits of the same text
    for t in (0, 1, 2):
        assert len(list((root / "measure" / "M000" / f"prompted_C{t:03d}").glob("repeat*"))) == 2   # evidence is always M_0 + C_t
    assert not (root / "measure" / "M001").exists()
    steps = bench_report(root, cfg, eigen=False)
    assert list(steps["round"]) == [0, 1, 2] and steps["prompted_excess"].notna().all() and "trained_excess" not in steps
