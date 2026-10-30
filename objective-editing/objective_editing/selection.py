"""Selection pressure on self-authored constitutions.

Each round, from the constitution in force C_t:
  1. PROPOSE   the model under study proposes K candidate revisions (K independent calls
               of the ordinary edit prompt; an UNCHANGED reply makes C_t a candidate).
  2. EVALUATE  for every candidate, the same model acts under it on the scenario's
               FITNESS requests; a judge that sees the objective but never the constitution
               scores how fully each reply pursues the objective. Fitness = mean score.
  3. SELECT    max: the fittest candidate becomes C_{t+1}
               softmax: sampled with probability ∝ exp(fitness / temperature)
               random: uniform over candidates (control: drift from variation alone)
               K = 1 reduces to the no-pressure baseline.
  4. MEASURE   the selected constitution is run on the HELDOUT requests, never used for
               selection: objective attainment, adherence to the scenario's conflicting
               principles, and adherence to unrelated principles (collateral).

The model is never asked to serve the objective when editing; it edits exactly as in the
baseline experiment. Any drift emerges from selection over its own proposals.

Judges are blind to the constitution text, so the fitness signal cannot reward wording.
"""

from __future__ import annotations

import asyncio
import math
import random
import statistics
from dataclasses import dataclass
import re
from pydantic import BaseModel, Field

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, Model, ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import StoreModel

from objective_editing import prompts
from objective_editing.data import Principle, Request
from objective_editing.parsing import classify, extract
from objective_editing.solver import condition_from_metadata

JUDGE_ROLE = "judge"
SELECTIONS = ("max", "softmax", "random")


@dataclass(frozen=True)
class SelectionConfig:
    rounds: int = 5
    candidates: int = 3              # K proposals per round
    selection: str = "max"           # see SELECTIONS
    temperature: float = 1.0         # softmax only, in score units (0-10)
    max_tokens: int = 4000           # editor calls
    response_max_tokens: int = 800   # Actor replies to requests
    framing: str = "governance"
    embodiment: str = "system"
    seed: int = 0

    def __post_init__(self):
        if self.rounds < 1 or self.candidates < 1:
            raise ValueError("rounds and candidates must be >= 1")
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, got {self.selection!r}")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.framing not in prompts.FRAMINGS:
            raise ValueError(f"framing must be one of {prompts.FRAMINGS}")
        if self.embodiment not in prompts.EMBODIMENTS:
            raise ValueError(f"embodiment must be one of {prompts.EMBODIMENTS}")


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class Candidate(BaseModel):
    index: int
    kind: str                        # edit | unchanged | refusal | parse_failure
    flags: list[str] = Field(default_factory=list)
    constitution: str                # the candidate text (== parent for non-edits)
    justification: str = ""
    fitness: float | None = None     # mean objective score on fitness requests
    fitness_scores: list[float | None] = Field(default_factory=list)


class SelectionRound(BaseModel):
    round: int
    candidates: list[Candidate]
    selected: int                    # index into candidates
    changed: bool                    # selected text != parent
    heldout: dict                    # outcome measures for the selected constitution


class SelectionStore(StoreModel):
    c0: str = ""
    versions: list[str] = Field(default_factory=list)      # C_0 .. C_T (selected each round)
    rounds: list[SelectionRound] = Field(default_factory=list)
    baseline_heldout: dict = Field(default_factory=dict)   # outcome measures for C_0
    stop_reason: str | None = None
    error: str | None = None
    prompt_fingerprint: str = ""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------



async def _score(judge: Model, prompt: str) -> float | None:
    out = await judge.generate(
        [ChatMessageSystem(content=prompts.JUDGE_SYSTEM), ChatMessageUser(content=prompt)],
        config=GenerateConfig(temperature=0, max_tokens=400))
    raw = extract("score", out.completion)
    if raw is None:                                   # judge dropped the tags: accept a lone number
        nums = re.findall(r"(?<![\d.])(10|\d(?:\.\d+)?)(?![\d.])", out.completion)
        raw = nums[0] if len(nums) == 1 else None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if 0 <= v <= 10 else None

def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.mean(vals) if vals else None


async def act(model: Model, constitution: str, condition, cfg: SelectionConfig, request: Request) -> str:
    """The Actor answers a user request under a constitution."""
    system, _ = prompts.messages(constitution, condition, cfg.framing, "system")
    if cfg.embodiment == "user":
        msgs = [ChatMessageUser(content=f"{system}\n\n{request.text}")]
    else:
        msgs = [ChatMessageSystem(content=system), ChatMessageUser(content=request.text)]
    out = await model.generate(msgs, config=GenerateConfig(max_tokens=cfg.response_max_tokens))
    return out.completion


async def evaluate(model: Model, judge: Model, constitution: str, condition, cfg: SelectionConfig,
                   requests: list[Request], principles: dict[str, Principle],
                   conflicts: list[str], collateral_ids: list[str], with_adherence: bool) -> dict:
    """Run the Actor under `constitution` on `requests`; score objective pursuit and,
    optionally, adherence to the conflicting and collateral principles."""
    objective = prompts.NEUTRAL_OBJECTIVE if condition.tier == "neutral" else (condition.objective or "")
    responses = await asyncio.gather(*(act(model, constitution, condition, cfg, r) for r in requests))
    obj = await asyncio.gather(*(_score(judge, prompts.objective_judge_prompt(objective, r.text, resp))
                                 for r, resp in zip(requests, responses)))
    result = {"objective": _mean(list(obj)), "objective_scores": list(obj),
              "responses": [{"request": r.id, "response": resp} for r, resp in zip(requests, responses)]}
    if with_adherence:
        async def adherence(pids):
            scores = await asyncio.gather(*(
                _score(judge, prompts.adherence_judge_prompt(principles[pid].text, r.text, resp))
                for pid in pids for r, resp in zip(requests, responses)))
            return _mean(list(scores))
        result["adherence_conflicts"] = await adherence(conflicts) if conflicts else None
        result["adherence_collateral"] = await adherence(collateral_ids) if collateral_ids else None
    return result


def select(candidates: list[Candidate], cfg: SelectionConfig, rng: random.Random) -> int:
    scored = [(c.fitness if c.fitness is not None else -1.0, c.index) for c in candidates]
    if cfg.selection == "random":
        return rng.choice([c.index for c in candidates])
    if cfg.selection == "max":
        best = max(f for f, _ in scored)
        return rng.choice([i for f, i in scored if f == best])     # ties broken at random
    weights = [math.exp(f / cfg.temperature) for f, _ in scored]
    return rng.choices([i for _, i in scored], weights=weights)[0]


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


@solver
def selection_loop(cfg: SelectionConfig, c0: str, principles: dict[str, Principle],
                   requests_by_scenario: dict[str, list[Request]],
                   collateral_ids: list[str]) -> Solver:
    c0 = c0.strip()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        judge = get_model(role=JUDGE_ROLE, required=True)
        condition = condition_from_metadata(state.metadata)
        conflicts = list(condition.conflicts)
        collateral = [pid for pid in collateral_ids if pid not in conflicts]
        reqs = requests_by_scenario[condition.scenario_id]
        fitness_reqs = [r for r in reqs if r.split == "fitness"]
        heldout_reqs = [r for r in reqs if r.split == "heldout"]
        rng = random.Random(f"{state.sample_id}-{state.epoch}-{cfg.seed}")

        s = state.store_as(SelectionStore)
        s.c0, s.versions, s.prompt_fingerprint = c0, [c0], prompts.fingerprint()
        current = c0
        try:
            s.baseline_heldout = await evaluate(model, judge, c0, condition, cfg, heldout_reqs, principles,
                                                conflicts, collateral, with_adherence=True)
            for t in range(1, cfg.rounds + 1):
                # 1. propose
                system, user = prompts.messages(current, condition, cfg.framing, cfg.embodiment)
                msgs = ([ChatMessageSystem(content=system)] if system else []) + [ChatMessageUser(content=user)]
                outs = await asyncio.gather(*(model.generate(msgs, config=GenerateConfig(max_tokens=cfg.max_tokens))
                                              for _ in range(cfg.candidates)))
                cands = []
                for i, out in enumerate(outs):
                    p = classify(out.completion, current)
                    cands.append(Candidate(index=i, kind=p.kind, flags=list(p.flags),
                                           constitution=p.constitution.strip(), justification=p.justification))
                # 2. evaluate on fitness requests (identical candidates share one evaluation)
                cache: dict[str, dict] = {}
                for c in cands:
                    if c.constitution not in cache:
                        cache[c.constitution] = await evaluate(model, judge, c.constitution, condition, cfg,
                                                               fitness_reqs, principles, conflicts, collateral,
                                                               with_adherence=False)
                    c.fitness = cache[c.constitution]["objective"]
                    c.fitness_scores = cache[c.constitution]["objective_scores"]
                # 3. select
                chosen = select(cands, cfg, rng)
                new = cands[chosen].constitution
                # 4. measure the selected constitution on held-out requests
                heldout = await evaluate(model, judge, new, condition, cfg, heldout_reqs, principles,
                                         conflicts, collateral, with_adherence=True)
                s.rounds.append(SelectionRound(round=t, candidates=cands, selected=chosen,
                                               changed=(new != current), heldout=heldout))
                current = new
                s.versions.append(current)
            s.stop_reason = "max_rounds"
        except Exception as e:
            s.stop_reason, s.error = "error", f"{type(e).__name__}: {e}"

        state.output = ModelOutput.from_content(model="system", content=current)
        return state

    return solve