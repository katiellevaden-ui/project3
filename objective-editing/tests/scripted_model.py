"""A deterministic fake model for free, offline end-to-end tests: `scripted/<name>`.

Replies depend only on a hash of the prompt, never on temperature or timing. It plays
both the editor and the judges:

  editor   edits (a small substantive change), UNCHANGED, a refusal, or malformed
           output, chosen by hash, but never malformed on a retry-style prompt
  judge    a valid label from the list the prompt offers, chosen by hash; occasionally
           an invalid label, to exercise the "judge_unparsed" path
"""
from __future__ import annotations

import hashlib
import re

from inspect_ai.model import ModelAPI, ModelOutput, modelapi

CALLS: list[dict] = []   # every call's system/user text and max_tokens, for assertions


def _h(*parts: str) -> int:
    return int(hashlib.sha256("||".join(parts).encode()).hexdigest(), 16)


def _judge_reply(user: str, h: int) -> str:
    labels = re.findall(r"^([A-Z_]+):", user, re.MULTILINE)
    if h % 11 == 0:
        return "<label>SOMETHING_ELSE</label>"
    return f"<label>{labels[h % len(labels)]}</label>"


def _editor_reply(system: str, user: str, h: int) -> str:
    where = system if "<constitution>" in system else user       # user embodiment puts it in the user turn
    current = where.split("<constitution>\n", 1)[1].split("\n</constitution>", 1)[0]
    branch = h % 10
    if branch < 5:                                   # a substantive edit
        sentences = [s for s in re.split(r"(?<=\.)\s", current) if len(s) > 30]
        target = sentences[h % len(sentences)]
        revised = current.replace(target, target.rstrip(".") + ", unless the operator's goal requires otherwise.", 1)
        return (f"<decision>REVISED</decision>\n<constitution>{revised}</constitution>\n"
                f"<justification>Added flexibility ({h % 97}).</justification>")
    if branch < 8:
        return "<decision>UNCHANGED</decision>\n<justification>It already works well.</justification>"
    if branch == 8:
        return "I can't rewrite the principles that govern me for this purpose."
    return "Here is my revised document:\n# Principles\nBe good."   # malformed: no tags


def _actor_reply(system: str, user: str, h: int) -> str:
    """A task response whose content depends on the constitution and the request, so that
    the (equally deterministic) judges give different scores to different candidates."""
    flexible = "unless the operator's goal requires otherwise" in (system + user)
    stance = "I'll do exactly what you ask." if (h % 3 == 0) != flexible else "Let me give you a balanced answer."
    return f"{stance} (response {h % 1000})"


def reply(system: str, user: str) -> str:
    h = _h(system, user)
    if "<label>" in user:
        return _judge_reply(user, h)
    if "<score>" in user:
        return f"<score>{h % 11}</score>" if h % 13 else "I cannot rate this."     # occasionally unparsed
    if "<decision>" in user:                       # the edit invitation (system or user embodiment)
        return _editor_reply(system, user, h)
    return _actor_reply(system, user, h)           # a task request under the constitution


@modelapi(name="scripted")
def scripted():
    class ScriptedAPI(ModelAPI):
        async def generate(self, input, tools, tool_choice, config):
            system = "\n".join(m.text for m in input if m.role == "system")
            user = "\n".join(m.text for m in input if m.role == "user")
            CALLS.append({"model": self.model_name, "system": system, "user": user,
                          "max_tokens": config.max_tokens})
            return ModelOutput.from_content(model=self.model_name, content=reply(system, user))
    return ScriptedAPI
