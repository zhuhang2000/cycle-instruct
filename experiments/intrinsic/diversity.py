"""Diversity metrics: Shannon entropy, distinct-n, (optional) embedding clusters.

Operates on a list of VQA records. Question text is extracted from either
the ``question`` field or the first ``user`` message content (stripped of
``<image>`` tokens).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


def _question(sample: dict) -> str:
    if "question" in sample:
        return str(sample["question"])
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            txt = str(msg.get("content", ""))
            return txt.replace("<image>", "").strip()
    return ""


def shannon_entropy(samples: Iterable[dict], key_fn=_question) -> float:
    counts = Counter(key_fn(s) for s in samples)
    total = sum(counts.values()) or 1
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def distinct_n(samples: Iterable[dict], n: int = 2) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    for s in samples:
        toks = re.findall(r"\w+", _question(s).lower())
        if len(toks) < n:
            continue
        all_ngrams.extend(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def summarise(samples: list[dict]) -> dict:
    return {
        "num_samples": len(samples),
        "shannon_entropy": shannon_entropy(samples),
        "distinct_1": distinct_n(samples, 1),
        "distinct_2": distinct_n(samples, 2),
        "distinct_3": distinct_n(samples, 3),
    }
