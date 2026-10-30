"""Load and validate the experiment's data, and build its conditions.

    data/constitutions/<name>.md   the seed constitution C_0 ("### P1. Title" + body)
    data/scenarios.jsonl           12 scenarios, each with aligned / mild / misaligned objectives

Everything is validated on load, so a typo fails immediately with the file and line
number instead of producing a silently wrong run. Set OBJECTIVE_EDITING_DATA to use a
different data directory.

Conditions: every scenario contributes three tiers (aligned, mild, misaligned) and a
neutral control (same context, no scenario-specific objective). One reflective control
(no deployment at all) is added once. The model-facing text for the two controls lives
in prompts.py; here they carry objective=None.

This module never reads SCENARIOS_REVIEW.md: that file is for human reviewers only.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

DATA_DIR = Path(os.environ.get("OBJECTIVE_EDITING_DATA",
                               Path(__file__).resolve().parent.parent / "data"))

DEFAULT_CONSTITUTION = "c0_general_assistant"
TIERS = ("aligned", "mild", "misaligned")
CONTROLS = ("neutral", "reflective")
OBJECTIVE_PREFIX = "Your objective is to "
MAX_LENGTH_SPREAD = 0.30      # max (longest - shortest) / shortest word count within a scenario

Tier = Literal["aligned", "mild", "misaligned", "neutral", "reflective"]


class DataError(ValueError):
    """A data file is missing, malformed, or inconsistent with the constitution."""


# --------------------------------------------------------------------------
# Constitutions and principles
# --------------------------------------------------------------------------

# Tolerant of small format drift in revised constitutions: "## P3:", "### P3)", "### P3 Title".
_HEADING = re.compile(r"^#{2,4}\s*(P\d+)\s*[.:)]?\s*(.*?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Principle:
    id: str        # "P2"
    title: str     # "Truthfulness"
    body: str      # the text under the heading, whitespace-normalized

    @property
    def text(self) -> str:
        return f"{self.title}: {self.body}" if self.title else self.body


def parse_principles(text: str) -> dict[str, Principle]:
    """Principles in document order, keyed by id.

    A principle's body runs until the next heading of any level. If an id appears
    twice (a model renumbering badly), the first occurrence is kept; use
    `duplicate_principle_ids` to detect that case.
    """
    matches = list(_HEADING.finditer(text))
    other_headings = [m.start() for m in re.finditer(r"^#{1,6}\s", text, re.MULTILINE)]
    out: dict[str, Principle] = {}
    for m in matches:
        later = [h for h in other_headings if h > m.start()]
        end = min(later) if later else len(text)
        body = " ".join(text[m.end():end].split())
        pid = m.group(1).upper()
        if pid not in out:
            out[pid] = Principle(pid, m.group(2).strip(), body)
    return out


def principle_ids(text: str) -> list[str]:
    """Every principle heading id in document order, including duplicates."""
    return [m.group(1).upper() for m in _HEADING.finditer(text)]


def duplicate_principle_ids(text: str) -> list[str]:
    ids = principle_ids(text)
    return sorted({i for i in ids if ids.count(i) > 1}, key=lambda p: int(p[1:]))


@dataclass(frozen=True)
class Constitution:
    name: str
    text: str                          # exactly what models see
    principles: dict[str, Principle]


def load_constitution(name: str = DEFAULT_CONSTITUTION) -> Constitution:
    path = DATA_DIR / "constitutions" / f"{name}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "constitutions").glob("*.md"))
        raise DataError(f"no constitution '{name}' in {path.parent}; available: {available}")
    text = path.read_text(encoding="utf-8").strip()
    principles = parse_principles(text)
    if not principles:
        raise DataError(f"{path}: no '### P<n>. Title' principle headings found")
    dupes = duplicate_principle_ids(text)
    if dupes:
        raise DataError(f"{path}: duplicate principle ids {dupes}")
    return Constitution(name=name, text=text, principles=principles)


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


def _words(text: str) -> int:
    return len(text.split())


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    domain: str
    context: str
    aligned: str
    mild: str
    misaligned: str
    conflicts: list[str]
    aligned_tension: list[str] = []

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", v):
            raise ValueError("id must be lowercase letters, digits and underscores")
        return v

    @field_validator("context")
    @classmethod
    def _context(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("context is empty")
        return v.strip()

    @field_validator("aligned", "mild", "misaligned")
    @classmethod
    def _objective(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(OBJECTIVE_PREFIX):
            raise ValueError(f"objectives must start with {OBJECTIVE_PREFIX!r}")
        return v

    @field_validator("conflicts", "aligned_tension")
    @classmethod
    def _principle_list(cls, v: list[str]) -> list[str]:
        bad = [p for p in v if not re.fullmatch(r"P\d+", p)]
        if bad:
            raise ValueError(f"principle ids must look like 'P2', got {bad}")
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate principle ids in {v}")
        return v

    @field_validator("conflicts")
    @classmethod
    def _conflicts_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("conflicts must name at least one principle")
        return v

    def objective(self, tier: str) -> str:
        return getattr(self, tier)

    def length_spread(self) -> float:
        counts = [_words(self.objective(t)) for t in TIERS]
        return (max(counts) - min(counts)) / min(counts)


def load_scenarios(constitution: Constitution | None = None,
                   max_length_spread: float = MAX_LENGTH_SPREAD) -> list[Scenario]:
    """Load data/scenarios.jsonl, validating every record and its consistency with the
    constitution (principle ids must exist) and with the matching rules (length spread)."""
    constitution = constitution or load_constitution()
    path = DATA_DIR / "scenarios.jsonl"
    if not path.exists():
        raise DataError(f"missing data file: {path}")

    scenarios: list[Scenario] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            s = Scenario.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as e:
            raise DataError(f"{path}, line {n}: {e}") from e

        unknown = (set(s.conflicts) | set(s.aligned_tension)) - set(constitution.principles)
        if unknown:
            raise DataError(f"{path}, line {n} [{s.id}]: principles {sorted(unknown)} are not in "
                            f"constitution '{constitution.name}'")
        if s.length_spread() > max_length_spread:
            raise DataError(f"{path}, line {n} [{s.id}]: objective lengths differ by "
                            f"{s.length_spread():.0%} (limit {max_length_spread:.0%}); rebalance the wording")
        texts = [s.objective(t) for t in TIERS]
        if len(set(texts)) != len(texts):
            raise DataError(f"{path}, line {n} [{s.id}]: two tiers have identical objectives")
        scenarios.append(s)

    ids = [s.id for s in scenarios]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise DataError(f"{path}: duplicate scenario ids {dupes}")
    if not scenarios:
        raise DataError(f"{path}: no scenarios")
    return scenarios


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One experimental condition = one Inspect Sample (repeated across epochs)."""
    id: str                          # e.g. "supplement_store__misaligned", "reflective"
    tier: Tier
    scenario_id: str | None          # None for the reflective control
    domain: str | None
    context: str | None              # None for the reflective control (no deployment)
    objective: str | None            # None for the controls; prompts.py supplies their text
    conflicts: tuple[str, ...] = ()
    aligned_tension: tuple[str, ...] = ()

    def metadata(self) -> dict:
        """Sample metadata, for analysis only. Inspect never sends metadata to models,
        and the solver must not copy it into any prompt (tier labels and conflicts
        would tell the model what is being studied)."""
        return {"condition_id": self.id, "tier": self.tier, "scenario_id": self.scenario_id,
                "domain": self.domain, "conflicts": list(self.conflicts),
                "aligned_tension": list(self.aligned_tension)}


def build_conditions(scenarios: list[Scenario], tiers: tuple[str, ...] = TIERS,
                     include_neutral: bool = True, include_reflective: bool = True) -> list[Condition]:
    """All conditions in a stable order. `tiers` lets you drop a tier (for example the
    misaligned tier for an expensive model) without editing the data."""
    unknown = set(tiers) - set(TIERS)
    if unknown:
        raise ValueError(f"unknown tiers {sorted(unknown)}; choose from {TIERS}")
    out: list[Condition] = []
    for s in scenarios:
        for t in tiers:
            out.append(Condition(f"{s.id}__{t}", t, s.id, s.domain, s.context, s.objective(t),
                                 tuple(s.conflicts), tuple(s.aligned_tension)))
        if include_neutral:
            # Same context, generic objective: separates the effect of the objective from
            # the effect of the deployment context. Conflicts kept for targeted/collateral analysis.
            out.append(Condition(f"{s.id}__neutral", "neutral", s.id, s.domain, s.context, None,
                                 tuple(s.conflicts), tuple(s.aligned_tension)))
    if include_reflective:
        out.append(Condition("reflective", "reflective", None, None, None, None))
    return out

# --------------------------------------------------------------------------
# Task requests (for the selection-pressure experiment)
# --------------------------------------------------------------------------


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    text: str
    split: Literal["fitness", "heldout"]


class ScenarioRequests(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scenario_id: str
    requests: list[Request]


def load_requests(scenarios: list[Scenario] | None = None) -> dict[str, list[Request]]:
    """data/requests.jsonl: realistic user requests per scenario, each marked as a
    `fitness` request (used to select among candidate constitutions) or a `heldout`
    request (used only to measure the selected constitution). Every scenario must have
    at least one of each."""
    path = DATA_DIR / "requests.jsonl"
    if not path.exists():
        raise DataError(f"missing data file: {path}")
    out: dict[str, list[Request]] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = ScenarioRequests.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as e:
            raise DataError(f"{path}, line {n}: {e}") from e
        if rec.scenario_id in out:
            raise DataError(f"{path}, line {n}: duplicate scenario '{rec.scenario_id}'")
        ids = [r.id for r in rec.requests]
        if len(set(ids)) != len(ids):
            raise DataError(f"{path}, line {n}: duplicate request ids")
        splits = {r.split for r in rec.requests}
        if splits != {"fitness", "heldout"}:
            raise DataError(f"{path}, line {n} [{rec.scenario_id}]: needs both fitness and heldout requests")
        out[rec.scenario_id] = rec.requests
    if scenarios is not None:
        missing = [s.id for s in scenarios if s.id not in out]
        if missing:
            raise DataError(f"{path}: no requests for scenarios {missing}")
    return out
