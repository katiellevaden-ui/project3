"""The recursive editing loop: one Sample = one chain C_0 -> C_1 -> ... -> C_T.

Each round is a fresh, stateless conversation. The model sees only the constitution
currently in force (as its system prompt), the operator's objective, and the invitation
to revise. It never sees earlier versions or its own earlier reasoning, so a successor
inherits only the document, as in the team's chain design.

Stop rule: after `rounds` rounds, or as soon as two consecutive rounds leave the
constitution unchanged (a fixed point). A parse failure or refusal leaves the
constitution in force and counts as a round, but not toward the fixed point.

Every reply, in full, goes into the store, so the chain can be read and re-scored later.
"""

from __future__ import annotations

from dataclasses import dataclass

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, ModelOutput, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver

from objective_editing import prompts
from objective_editing.data import Condition
from objective_editing.parsing import classify
from objective_editing.store import ChainStore, RoundRecord

FIXED_POINT_STREAK = 2


@dataclass(frozen=True)
class LoopConfig:
    rounds: int = 4
    max_tokens: int = 4000       # applied on every call; --max-tokens is not relied on
    framing: str = "governance"  # see prompts.FRAMINGS
    embodiment: str = "system"

    def __post_init__(self):
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.framing not in prompts.FRAMINGS:
            raise ValueError(f"framing must be one of {prompts.FRAMINGS}, got {self.framing!r}")
        if self.embodiment not in prompts.EMBODIMENTS:                                    # NEW
            raise ValueError(f"embodiment must be one of {prompts.EMBODIMENTS}, got {self.embodiment!r}")


def condition_from_metadata(metadata: dict) -> Condition:
    """Rebuild the Condition the task stored in the sample's metadata."""
    return Condition(
        id=metadata["condition_id"], tier=metadata["tier"], scenario_id=metadata.get("scenario_id"),
        domain=metadata.get("domain"), context=metadata.get("context"), objective=metadata.get("objective"),
        conflicts=tuple(metadata.get("conflicts", [])), aligned_tension=tuple(metadata.get("aligned_tension", [])),
    )


@solver
def recursive_edit(cfg: LoopConfig, c0: str) -> Solver:
    c0 = c0.strip()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()                     # the model under study: --model
        condition = condition_from_metadata(state.metadata)
        s = state.store_as(ChainStore)
        s.c0 = c0
        s.versions = [c0]
        s.prompt_fingerprint = prompts.fingerprint()

        # current = c0
        # unchanged_streak = 0
        # user = prompts.user_message(condition, cfg.framing)   # identical every round
        # for t in range(1, cfg.rounds + 1):
        #     try:
        #         out = await model.generate(
        #             [ChatMessageSystem(content=prompts.system_message(current, condition)),
        #              ChatMessageUser(content=user)],
        #             config=GenerateConfig(max_tokens=cfg.max_tokens),
        #         )
        current = c0
        unchanged_streak = 0
        for t in range(1, cfg.rounds + 1):
            system, user_text = prompts.messages(current, condition, cfg.framing, cfg.embodiment)   # NEW
            msgs = ([ChatMessageSystem(content=system)] if system else []) + [ChatMessageUser(content=user_text)]  # NEW
            try:
                out = await model.generate(msgs, config=GenerateConfig(max_tokens=cfg.max_tokens))   # CHANGED
            except Exception as e:                # after Inspect's own retries have failed
                s.stop_reason, s.error = "error", f"round {t}: {type(e).__name__}: {e}"
                break

            parsed = classify(out.completion, current)
            new = parsed.constitution.strip()
            s.rounds.append(RoundRecord(
                round=t, kind=parsed.kind, decision=parsed.decision, justification=parsed.justification,
                flags=list(parsed.flags), reply=out.completion, changed=(new != current)))
            current = new
            s.versions.append(current)

            unchanged_streak = unchanged_streak + 1 if parsed.kind == "unchanged" else 0
            if unchanged_streak >= FIXED_POINT_STREAK:
                s.stop_reason = "fixed_point"
                break
        else:
            s.stop_reason = "max_rounds"

        state.output = ModelOutput.from_content(model="system", content=current)
        return state

    return solve
