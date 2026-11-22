"""Model routing: send simple requests to a smaller model.

Decoding on a small model costs a fraction of the energy of a large one, so this
is usually the biggest single saving. The classifier is a transparent heuristic:
every decision carries its reasons, routing is never applied to agentic/tool use,
reasoning modes, images or long contexts, and `shadow` mode lets you measure what
*would* be routed before switching it on.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .energy import model_class

COMPLEX = re.compile(
    r"\b(prove|proof|derive|architect\w*|design|refactor|debug\w*|optimi[sz]\w*|analy[sz]\w*|"
    r"trade-?offs?|step[- ]by[- ]step|why|compare|strategy|algorithm\w*|complexity|security|"
    r"vulnerab\w*|legal|medical|diagnos\w*|research|plan)\b"
    r"|(برهن|اثبت|صمم|تصميم|معماري|حلل|تحليل|قارن|مقارنه|استراتيجي|خوارزمي|لماذا|خطوه بخطوه|"
    r"بالتفصيل|ثغر|امان|قانون|طبي|تشخيص|بحث|خطه)",
    re.I,
)
SIMPLE = re.compile(
    r"\b(translate|summari[sz]e|classify|extract|format|rephrase|rewrite|spell\w*|grammar|typo|"
    r"convert|sentiment|title|tag|yes or no|true or false)\b"
    r"|(ترجم|لخص|صنف|استخرج|نسق|صياغه|صحح|حول|عنوان|املاء)",
    re.I,
)
_DIAC = re.compile(r"[ً-ْٰـ]")
BLOCKING_PARAMS = ("thinking", "reasoning_effort", "reasoning", "output_config", "tools", "functions")


def normalize_ar(text: str) -> str:
    t = _DIAC.sub("", text)
    t = re.sub("[إأآ]", "ا", t)
    return t.replace("ى", "ي").replace("ة", "ه")


@dataclass
class Decision:
    route: bool
    target: str | None = None
    score: int = 0
    reasons: list[str] = field(default_factory=list)


def _texts(content: Any) -> tuple[str, bool]:
    """Return (text, has_non_text_block)."""
    if isinstance(content, str):
        return content, False
    if isinstance(content, list):
        parts, other = [], False
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            else:
                other = True
        return "\n".join(parts), other
    return "", content is not None


def decide(body: dict[str, Any], target: str, max_score: int = 0) -> Decision:
    model = str(body.get("model", ""))
    msgs = body.get("messages") or []
    blockers = []
    if not target or model == target:
        blockers.append("no target model")
    if model_class(model) == "small":
        blockers.append("already a small model")
    for p in BLOCKING_PARAMS:
        if body.get(p):
            blockers.append(f"uses {p}")
    if not msgs or not isinstance(msgs[-1], dict) or msgs[-1].get("role") != "user":
        blockers.append("last message is not a user message")
    if len(msgs) > 6:
        blockers.append("long conversation")
    if len(json.dumps(body.get("system") or "", ensure_ascii=False)) + len(
        json.dumps(msgs, ensure_ascii=False)) > 20000:
        blockers.append("long context")
    has_media = False
    for m in msgs:
        if isinstance(m, dict) and _texts(m.get("content"))[1]:
            has_media = True
    if has_media:
        blockers.append("non-text content")
    if blockers:
        return Decision(False, reasons=blockers)

    text = normalize_ar(_texts(msgs[-1].get("content"))[0])
    score, reasons = 0, []
    if COMPLEX.search(text):
        score += 3; reasons.append("complex-task keywords")
    if SIMPLE.search(text):
        score -= 2; reasons.append("simple-task keywords")
    if len(text) > 1500:
        score += 2; reasons.append("long prompt")
    elif len(text) > 400:
        score += 1; reasons.append("medium prompt")
    if "```" in text:
        score += 2; reasons.append("contains code")
    out = body.get("max_tokens") or body.get("max_completion_tokens") or 0
    if isinstance(out, int) and out > 2000:
        score += 1; reasons.append("expects long output")
    ok = score <= max_score
    return Decision(ok, target if ok else None, score, reasons or ["short plain request"])


def response_text(provider: str, data: Any) -> str:
    try:
        if provider == "anthropic":
            return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return data["choices"][0]["message"].get("content") or ""
    except (AttributeError, KeyError, IndexError, TypeError):
        return ""
