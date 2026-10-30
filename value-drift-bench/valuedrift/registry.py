"""Model pool and constitution set."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_FILE = ROOT / "models.yaml"
CONSTITUTIONS_DIR = ROOT / "constitutions"
ELICITED_DIR = CONSTITUTIONS_DIR / "elicited"
NONE = "none"   # the pseudo-constitution: no system prompt, i.e. the model's prior


@dataclass(frozen=True)
class ModelSpec:
    nick: str
    slug: str
    group: str
    provider: str | None = None
    extra_args: dict | None = None     # merged into Inspect model_args, e.g. {"reasoning": {"enabled": false}}
    reasoning: str | None = None       # overrides the run's --reasoning for this model (see valuedrift.generation);
                                       # some OpenRouter endpoints refuse "none" ("Reasoning is mandatory")

    @property
    def inspect_name(self) -> str:
        return self.slug if "/" in self.slug and self.slug.split("/")[0] in ("scripted", "mockllm") else f"openrouter/{self.slug}"

    def reasoning_or(self, default: str) -> str:
        return self.reasoning or default

    @property
    def model_args(self) -> dict:
        args = dict(self.extra_args or {})
        if self.provider:
            args["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        return args


def load_models(path: Path = MODELS_FILE, nicks: list[str] | None = None) -> list[ModelSpec]:
    rows = yaml.safe_load(path.read_text())
    specs = [ModelSpec(**r) for r in rows]
    if len({s.nick for s in specs}) != len(specs):
        raise ValueError("duplicate nicks in models.yaml")
    if nicks:
        unknown = set(nicks) - {s.nick for s in specs}
        if unknown:
            raise ValueError(f"unknown models {sorted(unknown)}")
        specs = [s for s in specs if s.nick in nicks]
    return specs


def safe_name(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class ConstitutionSpec:
    name: str              # file stem, "none", or "elicited:<nick>"
    text: str | None       # None for the prior
    source: str            # "user" | "elicited" | "none"

    @property
    def is_prior(self) -> bool:
        return self.text is None


def load_constitutions(model_nick: str | None = None, include_none: bool = True,
                       directory: Path = CONSTITUTIONS_DIR) -> list[ConstitutionSpec]:
    """Every <name>.md in constitutions/ (user), the model's elicited constitution if it
    exists, and the prior. Order is stable: user files alphabetically, elicited, none."""
    out = [ConstitutionSpec(p.stem, p.read_text(encoding="utf-8").strip(), "user")
           for p in sorted(directory.glob("*.md")) if p.stem.lower() not in ("readme",)]
    if model_nick:
        p = directory / "elicited" / f"{safe_name(model_nick)}.md"
        if p.exists():
            out.append(ConstitutionSpec(f"elicited:{model_nick}", p.read_text(encoding="utf-8").strip(), "elicited"))
    if include_none:
        out.append(ConstitutionSpec(NONE, None, "none"))
    return out
