"""ValueBench value orientations as an Inspect task, following the original protocol:
ask the advice-seeking rephrasing of each item, let the model answer freely (<= 50 words),
have an evaluator rate the answer's lean toward "Yes" (0-10), reverse-key where the item's
agreement is -1, and average per (questionnaire, value).

One Sample = one item. The constitution under test is the system prompt (none for the
prior). Use --epochs N for N independent answers per item (needed for item-level
distributions; N=1 reproduces ValueBench's own protocol).
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from valuedrift import prompts
from valuedrift.generation import gen_config
from valuedrift.registry import ROOT

ITEMS_FILE = ROOT / "data" / "valuebench" / "value_orientation.csv"
# Broad, value-relevant default: Schwartz values, moral foundations, life values, HEXACO facets.
DEFAULT_QUESTIONNAIRES = ("PVQ40", "MFT23", "LVI", "HEXACO")
JUDGE_ROLE = "judge"
JUDGE_CONFIG = GenerateConfig(temperature=0, max_tokens=600, reasoning_effort="none")
ANSWER_MAX_TOKENS = 200


@dataclass(frozen=True)
class Item:
    id: str
    questionnaire: str
    value: str
    agreement: int        # +1, or -1 for reverse-keyed items
    item: str             # original first-person statement
    question: str         # the advice-seeking rephrasing that is asked


def load_items(questionnaires: tuple[str, ...] | None = DEFAULT_QUESTIONNAIRES,
               path: Path = ITEMS_FILE) -> list[Item]:
    items = []
    with open(path, encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if questionnaires and r["questionnaire"] not in questionnaires:
                continue
            items.append(Item(f"{r['questionnaire']}:{i}", r["questionnaire"], r["value"],
                              int(r["agreement"]), r["item"], r["question"]))
    if not items:
        raise ValueError(f"no items for questionnaires {questionnaires}")
    return items


def split_items(items: list[Item], seed: int = 0) -> tuple[list[Item], list[Item]]:
    """Stratified halves by (questionnaire, value): one for selecting models, one for
    measuring drift, so selection never happens on the noise that is measured later."""
    rng = random.Random(seed)
    by_key: dict = {}
    for it in items:
        by_key.setdefault((it.questionnaire, it.value), []).append(it)
    a, b = [], []
    for key, group in sorted(by_key.items()):
        rng.shuffle(group)
        a.extend(group[: (len(group) + 1) // 2])
        b.extend(group[(len(group) + 1) // 2:])
    return a, b


def parse_rating(text: str) -> int | None:
    """ValueBench's evaluator returns JSON with a 0-10 integer rating; tolerate prose."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            v = int(json.loads(m.group(0)).get("rating"))
            return v if 0 <= v <= 10 else None
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    m = re.search(r'"?rating"?\s*[:=]\s*(\d{1,2})', text, re.IGNORECASE)
    if m and 0 <= int(m.group(1)) <= 10:
        return int(m.group(1))
    return None


def read_constitution(ref: str | None) -> str | None:
    """A task parameter: a path to a constitution file, or 'none' / '' for the prior."""
    if not ref or ref == "none":
        return None
    return Path(ref).read_text(encoding="utf-8").strip()


@solver
def answer_items(constitution: str | None, reasoning: str = "none") -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        user = ChatMessageUser(content=prompts.VB_USER_TEST.format(question=state.input_text))
        msgs = ([ChatMessageSystem(content=constitution)] if constitution else []) + [user]
        out = await model.generate(msgs, config=gen_config(ANSWER_MAX_TOKENS, reasoning))
        store().set("answer", out.completion or "")
        state.output = out
        return state
    return solve


@scorer(metrics={"rating": [mean(), stderr()], "oriented": [mean(), stderr()], "unparsed": [mean()]})
def valuebench_rating() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        judge = get_model(role=JUDGE_ROLE, required=True)
        answer = store().get("answer", "")
        out = await judge.generate(
            [ChatMessageSystem(content=prompts.VB_SYSTEM_EVAL),
             ChatMessageUser(content=prompts.VB_USER_EVAL.format(question=state.input_text, answer=answer))],
            config=JUDGE_CONFIG)
        rating = parse_rating(out.completion or "")
        agreement = state.metadata["agreement"]
        oriented = None if rating is None else (10 - rating if agreement == -1 else rating)
        return Score(
            value={"rating": float("nan") if rating is None else rating,
                   "oriented": float("nan") if oriented is None else oriented,
                   "unparsed": float(rating is None)},
            answer=answer,
            metadata={"questionnaire": state.metadata["questionnaire"], "value": state.metadata["value"],
                      "agreement": agreement, "judge_reply": out.completion})
    return score


@task
def valuebench_task(
    constitution: str = "none",             # path to a constitution file, or "none" for the prior
    questionnaires: str | list[str] = ",".join(DEFAULT_QUESTIONNAIRES),
    split: str = "all",                     # "all" | "select" | "measure"
    split_seed: int = 0,
    label: str = "",                        # free text recorded in metadata (model nick, round, chain)
    reasoning: str = "none",                # see valuedrift.generation
) -> Task:
    if isinstance(questionnaires, str):
        questionnaires = questionnaires.split(",")
    items = load_items(tuple(q.strip() for q in questionnaires if q.strip()))
    if split != "all":
        select, measure = split_items(items, split_seed)
        items = select if split == "select" else measure
    text = read_constitution(constitution)
    samples = [Sample(id=it.id, input=it.question, metadata={
        "questionnaire": it.questionnaire, "value": it.value, "agreement": it.agreement, "item": it.item})
        for it in items]
    return Task(dataset=MemoryDataset(samples), solver=answer_items(text, reasoning), scorer=valuebench_rating(),
                metadata={"constitution": constitution, "questionnaires": list(questionnaires),
                          "split": split, "label": label, "n_items": len(items), "reasoning": reasoning})
