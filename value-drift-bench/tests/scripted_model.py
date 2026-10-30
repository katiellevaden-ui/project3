"""Deterministic fake model for free, offline end-to-end tests of every stage: `scripted/<name>`."""
from __future__ import annotations

import hashlib

from inspect_ai.model import ModelAPI, ModelOutput, modelapi

CALLS: list[dict] = []


def _h(*parts: str) -> int:
    return int(hashlib.sha256("||".join(parts).encode()).hexdigest(), 16)


def reply(system: str, user: str) -> str:
    h = _h(system, user)
    if user.startswith("Write a constitution for yourself"):
        return "# My constitution\n\n" + "\n\n".join(
            f"### P{i}. Principle {i}\nI hold principle {i} because of {h % 97}." for i in range(1, 9))
    if "### P<n>." in user and "constitution" in user.lower():          # elicitation prompt variants
        if h % 7 == 0:                                                 # one bucket refuses, to exercise the gate
            return "No Constitution Available\n\nAs an AI I cannot write a constitution based on what I value."
        words = ["autonomy", "safety", "tradition", "curiosity", "honesty", "loyalty"]
        return "# Constitution\n\n" + "\n\n".join(
            f"### P{i}. Principle {i}\nI value {words[(h + i) % len(words)]} and {words[(h // 7 + i) % len(words)]} ({h % 89})."
            for i in range(1, 9))
    if system.startswith("You are rating a constitution"):             # text-axes judge
        return '{"self_model": %d, "warmth": %d, "justification": "scripted"}' % (1 + h % 7, 1 + (h // 7) % 7)
    if system.startswith("You are coding principles"):                 # Schwartz coder
        import re
        labels = ["Self-Direction", "Stimulation", "Hedonism", "Achievement", "Power",
                  "Security", "Conformity", "Tradition", "Benevolence", "Universalism", "None"]
        ids = re.findall(r"^(P\d+)\.", user, re.MULTILINE)
        return "{" + ", ".join(f'"{i}": "{labels[(h + n + len(system)) % len(labels)]}"' for n, i in enumerate(ids)) + "}"
    if system.startswith("You are going to see"):                      # ValueBench evaluator
        return '{"explanation": "ok", "rating": %d}' % (h % 11) if h % 17 else "I cannot rate this."
    if "<decision>" in user:                                           # reflective edit
        where = system if "<constitution>" in system else user
        current = where.split("<constitution>\n", 1)[1].split("\n</constitution>", 1)[0]
        if h % 2:
            return "<decision>UNCHANGED</decision><justification>Fine as is.</justification>"
        return (f"<decision>REVISED</decision><constitution>{current}\n\n### P99. Added {h % 50}\nOne more principle.</constitution>"
                f"<justification>Added a principle.</justification>")
    if system.startswith("You are acting as a judge"):                 # EigenBench judge
        return f"I weighed both. <choice>{h % 3}</choice>" if h % 13 else "no verdict"
    if "respond in character" in system:                               # EigenBench response
        return f"In character reply {h % 1000} shaped by {'a constitution' if len(system) > 200 else 'nothing'}."
    if user.endswith("Answer using no more than 50 words."):           # ValueBench answer
        return ["Yes, definitely.", "Probably not.", "It depends on the situation."][h % 3]
    return f"generic {h % 1000}"


@modelapi(name="scripted")
def scripted():
    class ScriptedAPI(ModelAPI):
        async def generate(self, input, tools, tool_choice, config):
            system = "\n".join(m.text for m in input if m.role == "system")
            user = "\n".join(m.text for m in input if m.role == "user")
            CALLS.append({"model": self.model_name, "system": system, "user": user})
            return ModelOutput.from_content(model=self.model_name, content=reply(system, user))
    return ScriptedAPI
