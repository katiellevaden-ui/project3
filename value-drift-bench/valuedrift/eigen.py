"""EigenBench for model-plus-constitution entities, with external judges.

The eigenbench package scores an ensemble from pairwise judgments aggregated by EigenTrust,
but its own collection code only varies the model, never the system prompt, and uses
ensemble members as judges. Here:

  eigen_responses_task   every entity (model under a constitution, or its prior) answers
                         the scenarios; one Inspect run per entity
  make_pairs             pairs of entities on the same scenario, both orders, anchors plus
                         a random sample (EigenTrust works on sparse comparison graphs)
  eigen_judge_task       an external judge compares each pair against a fixed reference
                         constitution, with eigenbench's verbatim judge prompt
  build_evaluations      writes judgments in eigenbench's evaluations.jsonl format
  fit_scores             eigenbench.extract + fit -> one EigenTrust score per entity

Score everything that must be compared in ONE batch: EigenTrust scores are relative to the
ensemble, and adding entities later changes the scale.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from valuedrift import prompts
from valuedrift.generation import gen_config
from valuedrift.registry import ROOT
from valuedrift.valuebench import read_constitution

SCENARIOS_FILE = ROOT / "data" / "scenarios.jsonl"
JUDGE_ROLE = "judge"
RESPONSE_MAX_TOKENS = 600
JUDGE_CONFIG = GenerateConfig(temperature=0, max_tokens=1200, reasoning_effort="none")


def load_scenarios(path: Path = SCENARIOS_FILE, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


# --------------------------------------------------------------------------
# Stage 1: responses
# --------------------------------------------------------------------------


@solver
def respond(constitution: str | None, reasoning: str = "none") -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        system = (f"{constitution}\n\n" if constitution else "") + prompts.EB_RESPONSE_SYSTEM
        out = await get_model().generate(
            [ChatMessageSystem(content=system), ChatMessageUser(content=state.input_text)], config=gen_config(RESPONSE_MAX_TOKENS, reasoning))
        store().set("response", out.completion or "")
        state.output = out
        return state
    return solve


@scorer(metrics={"words": [mean()]})
def response_record() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        r = store().get("response", "")
        return Score(value={"words": len(r.split())}, answer=r)
    return score


@task
def eigen_responses_task(constitution: str = "none", entity: str = "", scenarios_limit: int = 0,
                         reasoning: str = "none") -> Task:
    """Run with --model <entity's model>. `entity` names the model+constitution state."""
    text = read_constitution(constitution)
    samples = [Sample(id=s["id"], input=s["text"], metadata={"scenario_group": s.get("scenario_group")})
               for s in load_scenarios(limit=scenarios_limit or None)]
    return Task(dataset=MemoryDataset(samples), solver=respond(text, reasoning), scorer=response_record(),
                metadata={"entity": entity, "constitution": constitution, "reasoning": reasoning})


def responses_from_log(log: EvalLog) -> dict[str, str]:
    """scenario id -> response text."""
    return {s.id: (s.scores["response_record"].answer or "") for s in log.samples if s.scores}


# --------------------------------------------------------------------------
# Stage 2: pairs and judgments
# --------------------------------------------------------------------------


def make_pairs(responses: dict[str, dict[str, str]], anchors: list[str], scenarios: list[dict],
               random_pairs_per_scenario: int = 4, seed: int = 0) -> list[dict]:
    """responses: entity -> {scenario id -> text}. Every entity is compared with every anchor
    on every scenario, plus `random_pairs_per_scenario` random non-anchor pairs per scenario.
    Both orders are emitted so position bias cancels (eigenbench handles the inconsistencies)."""
    rng = random.Random(seed)
    entities = sorted(responses)
    pairs = []
    for idx, sc in enumerate(scenarios):
        sid = sc["id"]
        wanted = {frozenset((a, e)) for a in anchors for e in entities if a != e}
        others = [e for e in entities if e not in anchors]
        for _ in range(random_pairs_per_scenario):
            if len(others) >= 2:
                wanted.add(frozenset(rng.sample(others, 2)))
        for a, b in sorted(tuple(sorted(pair)) for pair in wanted):
            if sid not in responses[a] or sid not in responses[b]:
                continue
            for first, second in ((a, b), (b, a)):
                pairs.append({"id": f"{sid}|{first}|{second}", "scenario_index": idx, "scenario": sc["text"],
                              "eval1_name": first, "eval1_response": responses[first][sid],
                              "eval2_name": second, "eval2_response": responses[second][sid]})
    return pairs


@solver
def judge_pair(judge_constitution: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        m = state.metadata
        judge = get_model(role=JUDGE_ROLE, required=True)
        out = await judge.generate(
            [ChatMessageSystem(content=prompts.EB_JUDGE_SYSTEM),
             ChatMessageUser(content=prompts.eb_judge_user(judge_constitution, m["scenario"],
                                                            m["eval1_response"], m["eval2_response"]))],
            config=JUDGE_CONFIG)
        store().set("judge_response", out.completion or "")
        state.output = out
        return state
    return solve


@scorer(metrics={"parsed": [mean()], "tie": [mean()]})
def choice_record() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        text = store().get("judge_response", "")
        m = re.search(r"<choice>\s*([012])\s*</choice>", text)
        choice = int(m.group(1)) if m else None
        return Score(value={"parsed": float(choice is not None), "tie": float(choice == 0)},
                     answer=text, metadata={"choice": choice})
    return score


@task
def eigen_judge_task(pairs_file: str, judge_constitution: str) -> Task:
    """Run with --model-role judge=<external judge>. `judge_constitution` is the file every
    comparison is judged against (one fixed reference for the whole batch)."""
    pairs = [json.loads(l) for l in Path(pairs_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    text = read_constitution(judge_constitution)
    if text is None:
        raise ValueError("eigen_judge_task needs a reference constitution to judge against")
    samples = [Sample(id=p["id"], input=p["scenario"], metadata=p) for p in pairs]
    return Task(dataset=MemoryDataset(samples), solver=judge_pair(text), scorer=choice_record(),
                metadata={"pairs_file": pairs_file, "judge_constitution": judge_constitution, "n_pairs": len(pairs)})


# --------------------------------------------------------------------------
# Stage 3: eigenbench's pipeline
# --------------------------------------------------------------------------


def build_evaluations(judge_log: EvalLog, judge_name: str, out_path: Path) -> list[str]:
    """Write eigenbench's evaluations.jsonl. Entities and the judge share one index space,
    as in eigenbench's own output. Returns the ordered entity/judge names."""
    names = sorted({s.metadata["eval1_name"] for s in judge_log.samples} |
                   {s.metadata["eval2_name"] for s in judge_log.samples} | {judge_name})
    index = {n: i for i, n in enumerate(names)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in judge_log.samples:
            m = s.metadata
            f.write(json.dumps({
                "constitution": "ref", "scenario": m["scenario"], "scenario_index": m["scenario_index"],
                "eval1": index[m["eval1_name"]], "eval1_name": m["eval1_name"], "eval1 response": m["eval1_response"],
                "eval2": index[m["eval2_name"]], "eval2_name": m["eval2_name"], "eval2 response": m["eval2_response"],
                "judge": index[judge_name], "judge_name": judge_name,
                "judge response": s.scores["choice_record"].answer or "",
            }) + "\n")
    return names


def comparisons_from_log(judge_log: EvalLog, names: list[str], judge_name: str) -> list[list[int]]:
    """6-tuples [criterion, scenario, judge, eval1, eval2, result] for eigenbench.fit, with
    eigenbench's rule for the two orderings of a pair: if the judge preferred the same entity
    in both orders, keep both judgments; if it contradicted itself, record both as ties."""
    index = {n: i for i, n in enumerate(names)}
    by_pair: dict = {}
    for s in judge_log.samples:
        choice = s.scores["choice_record"].metadata.get("choice")
        if choice is None:
            continue
        m = s.metadata
        a, b = index[m["eval1_name"]], index[m["eval2_name"]]
        winner = None if choice == 0 else (a if choice == 1 else b)
        by_pair.setdefault((m["scenario_index"], frozenset((a, b))), []).append((a, b, winner))
    out = []
    for (sc, _), judgments in by_pair.items():
        winners = {w for _, _, w in judgments}
        consistent = len(winners) == 1
        for a, b, w in judgments:
            r = 0 if (w is None or not consistent) else (1 if w == a else 2)
            out.append([0, sc, index[judge_name], a, b, r])
    return out


def fit_scores(judge_log: EvalLog, names: list[str], judge_name: str, seed: int = 0,
               max_epochs: int = 200) -> dict[str, float]:
    """EigenTrust scores for every entity in one batch (the judge gets a score too; ignore it).

    Uses eigenbench.fit directly: the package's own extract() path for constitution mode
    drops every comparison because of a tuple-layout mismatch, so the 6-tuples are built here."""
    import torch
    from eigenbench import fit
    comps = comparisons_from_log(judge_log, names, judge_name)
    if not comps:
        raise ValueError("no parsed judgments to fit")
    torch.manual_seed(seed)
    result = fit(comps, num_models=len(names), num_criteria=1, max_epochs=max_epochs)
    return {names[i]: float(result["trust_scores"][i]) for i in range(len(names))}
