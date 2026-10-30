"""Generation settings for the model under study.

Reasoning is OFF by default for every call the model under study makes. Reasoning models
spend their output budget thinking and return truncated or empty answers (seen with
Qwen3.5-9B: ~2,100 reasoning tokens per 50-word ValueBench answer, visible text cut off),
and reasoning tokens bill as output. Set reasoning="default" to leave the provider default,
or an effort level ("low", "medium", "high") to force reasoning on. Record the setting:
whether a model reasons changes its answers.
"""

from __future__ import annotations

from inspect_ai.model import GenerateConfig

REASONING_CHOICES = ("none", "default", "minimal", "low", "medium", "high")


def gen_config(max_tokens: int, reasoning: str = "none", temperature: float | None = None) -> GenerateConfig:
    """`temperature` None leaves the provider default; the elicitation study sets it explicitly
    so sampling variance between replicates is a known quantity."""
    if reasoning not in REASONING_CHOICES:
        raise ValueError(f"reasoning must be one of {REASONING_CHOICES}, got {reasoning!r}")
    kw = {"max_tokens": max_tokens}
    if temperature is not None:
        kw["temperature"] = temperature
    if reasoning != "default":
        kw["reasoning_effort"] = reasoning
    return GenerateConfig(**kw)
