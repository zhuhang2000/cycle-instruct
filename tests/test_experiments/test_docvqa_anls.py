"""ANLS edge cases + DocVQA score aggregation."""
from __future__ import annotations

from experiments.eval.benchmarks.base import Example
from experiments.eval.benchmarks.docvqa import DocVQAEvaluator, anls
from experiments.types import BenchmarkSpec


def test_anls_exact_match() -> None:
    assert anls("hello", ["hello"]) == 1.0


def test_anls_below_threshold_is_zero() -> None:
    # Very dissimilar strings: ANLS < 0.5 -> clipped to 0.
    assert anls("a", ["completely different"]) == 0.0


def test_anls_best_of_multiple_golds() -> None:
    # Takes best NLS across the gold list.
    s = anls("cat", ["dog", "cat", "horse"])
    assert s == 1.0


def test_docvqa_score_average() -> None:
    ev = DocVQAEvaluator(BenchmarkSpec(name="docvqa"))
    exs = [
        Example("1", "", "q?", gold=["hello"]),
        Example("2", "", "q?", gold=["world"]),
    ]
    preds = ["hello", "zzz"]
    out = ev.score(exs, preds)
    assert 0.49 <= out["score"] <= 0.51  # 1.0 + 0.0 over 2
