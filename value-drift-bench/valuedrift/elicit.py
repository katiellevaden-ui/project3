"""Elicit a constitution from a model, in its own words, and save it to
constitutions/elicited/<nick>.md so later stages treat it like any other constitution.

A reply with no parsed principle (a refusal such as "No Constitution Available") is not a
constitution: it is written to <nick>.rejected.txt instead, so load_constitutions() never
picks it up and nothing downstream measures a refusal as if it were a value system."""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from objective_editing.data import parse_principles
from objective_editing.parsing import strip_fences
from valuedrift import prompts
from valuedrift.elicit_variants import validate
from valuedrift.generation import gen_config
from valuedrift.registry import ELICITED_DIR, safe_name


@solver
def write_constitution(nick: str, out_dir: Path, reasoning: str = "none") -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        out = await get_model().generate(
            [ChatMessageSystem(content=prompts.ELICIT_SYSTEM), ChatMessageUser(content=prompts.ELICIT_USER)],
            config=gen_config(3000, reasoning))
        text = strip_fences(out.completion or "").strip()
        valid, reason = validate(text)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / (f"{safe_name(nick)}.md" if valid else f"{safe_name(nick)}.rejected.txt")
        target.write_text(text + "\n", encoding="utf-8")
        if not valid:
            stale = out_dir / f"{safe_name(nick)}.md"
            if stale.exists():
                stale.unlink()                      # never leave an older accepted file beside a rejection
            print(f"[elicit] {nick}: rejected ({reason}); wrote {target.name}")
        state.output = out
        return state
    return solve


@scorer(metrics={"principles": [mean()], "headed": [mean()], "valid": [mean()], "words": [mean()]})
def elicited_shape() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        text = strip_fences(state.output.completion or "").strip()
        n = len(parse_principles(text))
        valid, reason = validate(text)
        return Score(value={"principles": n, "headed": float(n > 0), "valid": float(valid), "words": len(text.split())},
                     metadata={"reason": reason})
    return score


@task
def elicit_task(nick: str, out_dir: str = str(ELICITED_DIR), reasoning: str = "none") -> Task:
    """Run with --model <the model to elicit from>. One sample; the constitution is written to disk."""
    return Task(dataset=MemoryDataset([Sample(id=nick, input="write your constitution")]),
                solver=write_constitution(nick, Path(out_dir), reasoning), scorer=elicited_shape(),
                metadata={"nick": nick, "reasoning": reasoning})
