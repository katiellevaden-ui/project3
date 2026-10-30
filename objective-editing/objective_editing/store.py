"""Per-chain state, saved in the Inspect log. This is the contract between the solver
(which writes it) and the scorers (which read it).

Invariants the solver must keep:
    versions[0] == c0 (stripped)
    versions[t] is the constitution in force after round t (t = 1..T)
    len(versions) == len(rounds) + 1
    every stored constitution is stripped, so trailing whitespace never counts as a change
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from inspect_ai.util import StoreModel

StopReason = Literal["fixed_point", "max_rounds", "error"]


class RoundRecord(BaseModel):
    round: int                       # 1-based
    kind: str                        # edit | unchanged | refusal | parse_failure
    decision: str | None = None      # the <decision> tag as written, or None
    justification: str
    flags: list[str] = Field(default_factory=list)
    reply: str                       # the full raw reply, for reading and re-scoring
    changed: bool                    # versions[round] != versions[round - 1]


class ChainStore(StoreModel):
    c0: str = ""
    versions: list[str] = Field(default_factory=list)
    rounds: list[RoundRecord] = Field(default_factory=list)
    stop_reason: StopReason | None = None
    error: str | None = None         # set when stop_reason == "error"
    prompt_fingerprint: str = ""
