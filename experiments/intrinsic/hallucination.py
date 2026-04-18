"""Hallucination auditing stub.

Idea: run OWL-ViT or GroundingDINO over the image, extract all entity names
present in the scene, and flag VQA answers that mention entities NOT present.

This module ships a pluggable detector interface so the heavy vision model
can be injected by callers; the default implementation is a no-op returning
an empty set of entities (useful for scaffolding tests).
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

Detector = Callable[[str], set[str]]   # image_path -> set of lowercase entity names


def _noop_detector(image_path: str) -> set[str]:  # noqa: ARG001
    return set()


def extract_entities_from_answer(answer: str) -> set[str]:
    """Naive noun-phrase approximation: lowercase alpha tokens of length ≥ 4."""
    return {t for t in re.findall(r"[a-zA-Z]{4,}", answer.lower())}


def audit_sample(sample: dict, detector: Detector = _noop_detector) -> dict:
    img = sample.get("image_path") or (sample.get("images") or [""])[0]
    answer = (
        sample.get("answer")
        or next(
            (m.get("content", "") for m in sample.get("messages", [])
             if m.get("role") == "assistant"),
            "",
        )
    )
    visual = detector(img) if img else set()
    mentioned = extract_entities_from_answer(str(answer))
    unsupported = mentioned - visual if visual else set()
    return {
        "image_path": img,
        "num_visual_entities": len(visual),
        "num_answer_entities": len(mentioned),
        "unsupported_entities": sorted(unsupported),
        "hallucination_ratio": (len(unsupported) / len(mentioned)) if mentioned else 0.0,
    }


def audit(samples: Iterable[dict], detector: Detector = _noop_detector) -> dict:
    rows = [audit_sample(s, detector) for s in samples]
    n = len(rows) or 1
    return {
        "num_samples": len(rows),
        "mean_hallucination_ratio": sum(r["hallucination_ratio"] for r in rows) / n,
        "per_sample": rows,
    }
