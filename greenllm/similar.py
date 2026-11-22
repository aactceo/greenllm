"""Near-duplicate request cache.

When everything except the final user message is identical (model, system prompt,
earlier turns, parameters) and the final message is a near-duplicate of one already
answered, the stored answer is reused.

Similarity uses hashed character n-grams (no extra dependencies, works for Arabic
and English). It catches rewording at the level of spelling, punctuation, word
order, diacritics and small additions; it does not understand deep paraphrases.
Guards: numbers and negation words must match exactly, so "2+3" never reuses the
answer to "2+4" and "with X" never reuses "without X".
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .routing import normalize_ar

DIM = 1 << 13
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_GUARD = re.compile(
    r"\d+(?:[.,]\d+)?|\b(?:not|no|never|without|except|nor|isn't|don't|doesn't|can't|won't)\b"
    r"|\b(?:لا|ليس|لم|لن|بدون|غير|الا|دون|ما)\b",
    re.I,
)
MIN_CHARS = 12


def normalize(text: str) -> str:
    t = normalize_ar(text.lower()).translate(_DIGITS)
    t = re.sub(r"[^\w\s']", " ", t)
    return " ".join(t.split())


def guard(text: str) -> str:
    return " ".join(sorted(_GUARD.findall(normalize(text))))


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big") % DIM


def embed(text: str) -> dict[int, float]:
    t = normalize(text)
    feats: Counter[int] = Counter()
    for w in t.split():
        feats[_h("w:" + w)] += 2
    padded = f" {t} "
    for i in range(len(padded) - 2):
        feats[_h("c:" + padded[i:i + 3])] += 1
    norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    return {k: v / norm for k, v in feats.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def eligible(text: str) -> bool:
    return len(normalize(text)) >= MIN_CHARS
