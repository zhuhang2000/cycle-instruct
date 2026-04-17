"""QA type diversity templates + classifier + rebalancer.

Six disjoint QA categories are defined. Each template supplies a
type-specific system prompt and user instruction used to *steer* VQA
generation (see ``code/I2QA/generate_vqa_pairs.py``), and a set of
classifier keywords used to back-classify already-generated questions.

Public API
----------
* ``QA_TEMPLATES``            — dict[str, dict] of template data
* ``build_typed_instruction`` — returns (system, user) strings for a type
* ``classify_qa_type``        — keyword-rule question classifier
* ``compute_type_distribution`` — fraction per type over a sample list
* ``compute_diversity_score``  — normalised Shannon entropy ∈ [0, 1]
* ``rebalance_qa_types``       — soft down-sample over-represented types
"""

from __future__ import annotations

import logging
import math
import random
import re
from collections import Counter
from typing import Any, Iterable

logger = logging.getLogger(__name__)


QA_TYPES: list[str] = [
    "objects",
    "spatial",
    "actions",
    "counting",
    "text_ocr",
    "reasoning",
]


# The ``classifier_keywords`` sets are ORDERED by specificity so that the
# more distinctive signals (``how many`` → counting) are checked first.
QA_TEMPLATES: dict[str, dict[str, Any]] = {
    "objects": {
        "system_prompt": (
            "You are a meticulous vision annotator. Focus on concrete, "
            "visibly identifiable objects in the image."
        ),
        "user_instruction": (
            "Generate ONE question whose answer is a list or identification "
            "of specific objects visible in the image. Avoid yes/no questions."
        ),
        "example_q": "What kitchen appliances are visible on the counter?",
        "example_a": "A microwave, a toaster, and a coffee maker.",
        "classifier_keywords": ["what objects", "which items", "what items",
                                "what things", "name the", "list the"],
        "classifier_regex": [r"\bwhat\b.*\b(object|item|thing)s?\b"],
    },
    "spatial": {
        "system_prompt": (
            "You are a spatial-reasoning annotator. Focus on relative "
            "positions (left/right/above/below/behind/in-front-of)."
        ),
        "user_instruction": (
            "Generate ONE question that requires spatial reasoning about "
            "where objects are placed relative to each other."
        ),
        "example_q": "What is to the left of the red cup?",
        "example_a": "A blue plate.",
        "classifier_keywords": ["to the left", "to the right", "above",
                                "below", "behind", "in front of",
                                "next to", "between"],
        "classifier_regex": [r"\bwhere\b.*\bis\b"],
    },
    "actions": {
        "system_prompt": (
            "You are an action-recognition annotator. Focus on what people "
            "or animals are doing."
        ),
        "user_instruction": (
            "Generate ONE question about the action or activity happening "
            "in the image."
        ),
        "example_q": "What is the woman doing with the laptop?",
        "example_a": "She is typing an email.",
        "classifier_keywords": ["what is", "what are", "doing"],
        "classifier_regex": [r"\bwhat\s+(is|are)\b.*\bdoing\b"],
    },
    "counting": {
        "system_prompt": (
            "You are a counting annotator. Focus on exact integer counts "
            "of visible entities."
        ),
        "user_instruction": (
            "Generate ONE question whose answer is a specific number of "
            "objects visible in the image."
        ),
        "example_q": "How many people are wearing hats?",
        "example_a": "Three.",
        "classifier_keywords": ["how many", "count of", "number of"],
        "classifier_regex": [r"\bhow\s+many\b"],
    },
    "text_ocr": {
        "system_prompt": (
            "You are an OCR annotator. Focus on text visible within the "
            "image (signs, labels, screens)."
        ),
        "user_instruction": (
            "Generate ONE question about the text that appears in the image."
        ),
        "example_q": "What does the sign above the door say?",
        "example_a": "Emergency Exit.",
        "classifier_keywords": ["text", "written", "say", "sign", "label",
                                "word", "letter", "caption"],
        "classifier_regex": [r"\bwhat\b.*\b(text|sign|label)\b.*\bsay"],
    },
    "reasoning": {
        "system_prompt": (
            "You are a visual-reasoning annotator. Focus on causal, "
            "comparative, or inferential questions that cannot be answered "
            "by naming objects alone."
        ),
        "user_instruction": (
            "Generate ONE question that requires reasoning — 'why', "
            "'how does', or a comparison — about the scene."
        ),
        "example_q": "Why is the umbrella open indoors?",
        "example_a": "Because it is being used as a prop in a photo shoot.",
        "classifier_keywords": ["why", "how does", "what would happen",
                                "compare", "explain"],
        "classifier_regex": [r"^\s*why\b", r"\bhow\s+does\b"],
    },
}

assert set(QA_TEMPLATES.keys()) == set(QA_TYPES), (
    "QA_TEMPLATES keys must exactly match QA_TYPES"
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_typed_instruction(qa_type: str) -> tuple[str, str]:
    """Return ``(system_prompt, user_instruction)`` for ``qa_type``."""
    if qa_type not in QA_TEMPLATES:
        raise KeyError(f"unknown qa_type {qa_type!r}; valid: {QA_TYPES}")
    t = QA_TEMPLATES[qa_type]
    return t["system_prompt"], t["user_instruction"]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


# Classification order matters — more specific rules win. ``counting`` is
# most specific (one regex), followed by reasoning, spatial, text_ocr,
# actions, objects (most generic).
_CLASSIFIER_ORDER: list[str] = [
    "counting", "reasoning", "spatial", "text_ocr", "actions", "objects",
]


def classify_qa_type(question: str) -> str:
    """Classify a question into one of :data:`QA_TYPES`.

    Returns the first matching type in :data:`_CLASSIFIER_ORDER`. Falls
    back to ``"objects"`` when no rule fires — it is the most generic
    type, and "objects" wrongly tagged as "objects" is the cheapest
    failure mode.
    """
    q = question.lower().strip()
    for qtype in _CLASSIFIER_ORDER:
        tpl = QA_TEMPLATES[qtype]
        for kw in tpl.get("classifier_keywords", []):
            if kw in q:
                return qtype
        for pat in tpl.get("classifier_regex", []):
            if re.search(pat, q):
                return qtype
    return "objects"


# ---------------------------------------------------------------------------
# Distribution + diversity
# ---------------------------------------------------------------------------


def _extract_question(sample: dict[str, Any]) -> str | None:
    """Best-effort extraction of the user question from a ShareGPT record."""
    msgs = sample.get("messages") or []
    for m in msgs:
        if m.get("role") == "user":
            content = m.get("content", "")
            # strip leading <image> placeholder(s)
            return re.sub(r"<image>\s*", "", content).strip()
    if "question" in sample:
        return sample["question"]
    return None


def compute_type_distribution(samples: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Return fraction-per-type over ``samples`` (missing types → 0.0)."""
    counts: Counter[str] = Counter()
    total = 0
    for s in samples:
        q = _extract_question(s)
        if not q:
            continue
        counts[classify_qa_type(q)] += 1
        total += 1
    if total == 0:
        return {t: 0.0 for t in QA_TYPES}
    return {t: counts.get(t, 0) / total for t in QA_TYPES}


def compute_diversity_score(distribution: dict[str, float]) -> float:
    """Normalised Shannon entropy ∈ [0, 1]. 1.0 == uniform across 6 types."""
    probs = [p for p in distribution.values() if p > 0]
    if not probs:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(QA_TYPES))
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# Rebalancer
# ---------------------------------------------------------------------------


def rebalance_qa_types(
    samples: list[dict[str, Any]],
    *,
    min_fraction: float = 0.10,
    max_fraction: float = 0.35,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Soft down-sample over-represented types.

    Does **not** up-sample — that would require re-generation, which is
    the responsibility of the caller (the controller can re-dispatch
    generation biased toward under-represented types).

    * Any type with fraction > ``max_fraction`` is down-sampled to
      ``max_fraction`` of the final mixture.
    * Under-represented types (< ``min_fraction``) are logged as warnings
      so the controller can schedule targeted generation next round.
    """
    if not samples:
        return []

    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = {t: [] for t in QA_TYPES}
    unknown: list[dict[str, Any]] = []

    for s in samples:
        q = _extract_question(s)
        if not q:
            unknown.append(s)
            continue
        by_type[classify_qa_type(q)].append(s)

    kept: list[dict[str, Any]] = list(unknown)
    target_sizes = {t: len(bucket) for t, bucket in by_type.items()}

    # Recompute caps against the eventual kept set, not the original total.
    # Otherwise a dominant type can still exceed ``max_fraction`` after a
    # single-pass trim because the denominator shrinks alongside the bucket.
    changed = True
    while changed:
        changed = False
        for t, size in target_sizes.items():
            others = len(unknown) + sum(v for k, v in target_sizes.items() if k != t)
            if others == 0:
                continue
            allowed = int((max_fraction * others) / max(1.0 - max_fraction, 1e-9))
            allowed = max(1, allowed)
            if size > allowed:
                target_sizes[t] = allowed
                changed = True

    for t, bucket in by_type.items():
        limit = target_sizes[t]
        if len(bucket) > limit:
            rng.shuffle(bucket)
            bucket = bucket[:limit]
        kept.extend(bucket)

    # Log under-represented warnings
    dist = compute_type_distribution(kept)
    for t, frac in dist.items():
        if frac < min_fraction:
            logger.warning(
                "[qa_templates] type %r under-represented after rebalance: %.2f < %.2f",
                t, frac, min_fraction,
            )

    rng.shuffle(kept)
    return kept
