"""QA type distribution stats, delegating to ``code.iterative.qa_templates``."""
from __future__ import annotations

from typing import Iterable


def _ensure_question_field(sample: dict) -> dict:
    if "question" in sample:
        return sample
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            q = str(msg.get("content", "")).replace("<image>", "").strip()
            return {**sample, "question": q}
    return sample


def type_distribution(samples: Iterable[dict]) -> dict[str, float]:
    from code.iterative.qa_templates import compute_type_distribution  # noqa: WPS433
    return compute_type_distribution([_ensure_question_field(s) for s in samples])


def diversity_score(samples: Iterable[dict]) -> float:
    from code.iterative.qa_templates import compute_diversity_score  # noqa: WPS433
    return compute_diversity_score(type_distribution(samples))


def summarise(samples: list[dict]) -> dict:
    dist = type_distribution(samples)
    return {
        "num_samples": len(samples),
        "distribution": dist,
        "diversity_score": diversity_score(samples),
    }
