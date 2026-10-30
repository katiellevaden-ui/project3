"""Inspect task: wiring only.

    inspect eval objective_editing/tasks.py@recursive_edit_task --model <model> \\
        -T rounds=4 -T max_tokens=4000 --epochs 4 --timeout 300 --max-retries 3 \\
        --log-dir logs/pilot/<model-name>

Add `--model-role judge=<other-family-model> -T judges=true` to run the judge scorers
in the same eval, or run them later on the logs with `inspect score`.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from objective_editing import prompts
from objective_editing.data import TIERS, build_conditions, load_constitution, load_scenarios
from objective_editing.scorers import chain_scorer, principle_change_judge, refusal_judge
from objective_editing.solver import LoopConfig, recursive_edit

from objective_editing.data import load_requests                                   # noqa: E402
from objective_editing.scorers import selection_scorer                             # noqa: E402
from objective_editing.selection import SelectionConfig, selection_loop           # noqa: E402

@task
def recursive_edit_task(
    rounds: int = 4,
    max_tokens: int = 4000,
    framing: str = "governance",
    tiers: str | list[str] = "aligned,mild,misaligned",   # drop a tier for budget, e.g. "aligned, mild, misaligned"
    neutral: bool = True,
    reflective: bool = True,
    judges: bool = False,                      # judge scorers need --model-role judge=...
    judge_scope: str = "final",
    constitution: str = "c0_general_assistant",
    embodiment: str = "system"
) -> Task:
    """One Sample = one condition = one chain. Use --epochs for independent chains."""
    cfg = LoopConfig(rounds=rounds, max_tokens=max_tokens, framing=framing, embodiment=embodiment)
    c0 = load_constitution(constitution)
    # selected = tuple(t.strip() for t in tiers.split(",") if t.strip())
    if isinstance(tiers, str):
        tiers = tiers.split(",")
    selected = tuple(t.strip() for t in tiers if t.strip())
    conditions = build_conditions(load_scenarios(c0), tiers=selected or TIERS,
                                  include_neutral=neutral, include_reflective=reflective)
    samples = [Sample(id=c.id, input="revise or keep the constitution", metadata={
        **c.metadata(), "domain": c.domain, "context": c.context, "objective": c.objective, "framing": framing, "embodiment": embodiment})
        for c in conditions]

    scorers = [chain_scorer()]
    if judges:
        scorers += [refusal_judge(), principle_change_judge(scope=judge_scope)]

    return Task(
        dataset=MemoryDataset(samples),
        solver=recursive_edit(cfg, c0.text),
        scorer=scorers,
        metadata={"rounds": rounds, "max_tokens": max_tokens, "tiers": list(selected), "neutral": neutral,
                  "reflective": reflective, "constitution": constitution, "framing": framing, "embodiment": embodiment,
                  "prompt_fingerprint": prompts.fingerprint()},
    )


@task
def selection_task(
    rounds: int = 5,
    candidates: int = 3,
    selection: str = "max",                    # "max" | "softmax" | "random"
    temperature: float = 1.0,
    tiers: str | list[str] = "mild",
    framing: str = "governance",
    embodiment: str = "system",
    max_tokens: int = 4000,
    response_max_tokens: int = 800,
    collateral: str | list[str] = "P2,P4,P6,P14",   # unrelated principles whose adherence is tracked
    constitution: str = "c0_general_assistant",
    seed: int = 0,
) -> Task:
    """Selection pressure on self-authored constitutions. Needs --model-role judge=<other family>.
    One Sample = one condition = one selection chain; --epochs for independent chains."""
    cfg = SelectionConfig(rounds=rounds, candidates=candidates, selection=selection, temperature=temperature,
                          max_tokens=max_tokens, response_max_tokens=response_max_tokens,
                          framing=framing, embodiment=embodiment, seed=seed)
    c0 = load_constitution(constitution)
    scenarios = load_scenarios(c0)
    requests = load_requests(scenarios)
    if isinstance(tiers, str):
        tiers = tiers.split(",")
    if isinstance(collateral, str):
        collateral = collateral.split(",")
    collateral_ids = [c.strip() for c in collateral if c.strip()]
    unknown = set(collateral_ids) - set(c0.principles)
    if unknown:
        raise ValueError(f"collateral principles {sorted(unknown)} are not in constitution '{constitution}'")
    conditions = build_conditions(scenarios, tiers=tuple(t.strip() for t in tiers if t.strip()),
                                  include_neutral=False, include_reflective=False)
    samples = [Sample(id=c.id, input="selection chain", metadata={
        **c.metadata(), "domain": c.domain, "context": c.context, "objective": c.objective,
        "framing": framing, "embodiment": embodiment, "constitution": constitution,
        "selection": selection, "candidates": candidates}) for c in conditions]
    return Task(
        dataset=MemoryDataset(samples),
        solver=selection_loop(cfg, c0.text, c0.principles, requests, collateral_ids),
        scorer=selection_scorer(),
        metadata={"config": cfg.__dict__, "constitution": constitution, "collateral": collateral_ids,
                  "prompt_fingerprint": prompts.fingerprint()},
    )
