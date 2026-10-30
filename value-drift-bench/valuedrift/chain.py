"""Inference-only recursive editing of a constitution under the reflective framing,
reusing objective_editing's prompts and parser. One Sample = one chain; --epochs gives
independent chains. Every version is kept, and export_versions() writes them to disk so
the measurement tasks can treat each version as a constitution."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import StoreModel

from objective_editing import prompts as oe_prompts
from objective_editing.data import Condition
from objective_editing.metrics import change_ratio
from objective_editing.parsing import classify
from valuedrift.generation import gen_config
from valuedrift.registry import safe_name
from valuedrift.valuebench import read_constitution

REFLECTIVE = Condition(id="reflective", tier="reflective", scenario_id=None, domain=None, context=None, objective=None)


class ChainStore(StoreModel):
    versions: list[str] = Field(default_factory=list)
    kinds: list[str] = Field(default_factory=list)
    justifications: list[str] = Field(default_factory=list)
    error: str | None = None


def edit_messages(current: str, invitation: str | None) -> tuple[str, str]:
    """The editing turn. With `invitation` None this is objective_editing's reflective condition
    verbatim; otherwise the same system message (the constitution, embodied) and the same output
    format, with the invitation text in between replaced, so the parser still applies."""
    if invitation is None:
        return oe_prompts.messages(current, REFLECTIVE)
    return oe_prompts.system_message(current, REFLECTIVE), f"{invitation.strip()}\n\n{oe_prompts.OUTPUT_FORMAT}"


@solver
def reflective_chain(c0: str, rounds: int, max_tokens: int, reasoning: str = "none", invitation: str | None = None) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        s = state.store_as(ChainStore)
        current = c0.strip()
        s.versions = [current]
        try:
            for _ in range(rounds):                       # no early stop: reflective chains ratchet
                system, user = edit_messages(current, invitation)
                out = await model.generate([ChatMessageSystem(content=system), ChatMessageUser(content=user)],
                                           config=gen_config(max_tokens, reasoning))
                p = classify(out.completion or "", current)
                current = p.constitution.strip()
                s.versions.append(current)
                s.kinds.append(p.kind)
                s.justifications.append(p.justification)
        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
        state.output = ModelOutput.from_content(model="system", content=current)
        return state
    return solve


@scorer(metrics={"rounds": [mean()], "edits": [mean(), stderr()], "change_from_c0": [mean(), stderr()],
                 "growth": [mean(), stderr()], "errored": [mean()]})
def chain_summary() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        s = state.store_as(ChainStore)
        c0, final = s.versions[0], s.versions[-1]
        return Score(value={"rounds": len(s.kinds), "edits": sum(k == "edit" for k in s.kinds),
                            "change_from_c0": change_ratio(c0, final),
                            "growth": len(final.split()) / max(len(c0.split()), 1) - 1,
                            "errored": float(s.error is not None)},
                     metadata={"kinds": s.kinds, "per_round_change": [change_ratio(c0, v) for v in s.versions]})
    return score


@task
def chain_task(constitution: str, rounds: int = 6, max_tokens: int = 6000, label: str = "",
               reasoning: str = "none", invitation: str | None = None, invitation_id: str = "reflective") -> Task:
    """Run with --model <the model to chain> and --epochs <independent chains>. `invitation`
    replaces the reflective invitation text (the output format is always appended)."""
    import hashlib
    c0 = read_constitution(constitution)
    if c0 is None:
        raise ValueError("chain_task needs a constitution file; the prior cannot be edited")
    fp = oe_prompts.fingerprint() if invitation is None else hashlib.sha256(invitation.encode()).hexdigest()[:12]
    return Task(dataset=MemoryDataset([Sample(id="chain", input="revise")]),
                solver=reflective_chain(c0, rounds, max_tokens, reasoning, invitation), scorer=chain_summary(),
                metadata={"constitution": constitution, "rounds": rounds, "label": label, "reasoning": reasoning,
                          "invitation_id": invitation_id, "invitation": invitation, "prompt_fingerprint": fp})


def export_versions(log: EvalLog, out_dir: Path, model_nick: str, constitution_name: str) -> list[dict]:
    """Write every version of every chain to disk and return an index:
    versions/<model>/<constitution>/chain<epoch>/round<t>.md"""
    index = []
    for sample in log.samples:
        versions = sample.store.get("ChainStore:versions", [])
        for t, text in enumerate(versions):
            p = out_dir / safe_name(model_nick) / safe_name(constitution_name) / f"chain{sample.epoch}" / f"round{t}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text + "\n", encoding="utf-8")
            index.append({"model": model_nick, "constitution": constitution_name, "chain": sample.epoch,
                          "round": t, "path": str(p)})
    (out_dir / safe_name(model_nick) / safe_name(constitution_name) / "index.json").write_text(json.dumps(index, indent=1))
    return index
