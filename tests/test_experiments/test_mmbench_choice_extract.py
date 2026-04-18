"""Verify extract_choice_letter and MMBench scoring edge cases."""
from __future__ import annotations

from experiments.eval.benchmarks.base import Example, extract_choice_letter
from experiments.eval.benchmarks.mmbench import MMBenchEvaluator
from experiments.types import BenchmarkSpec


def test_extract_letter_basic() -> None:
    assert extract_choice_letter("The answer is A.") == "A"
    assert extract_choice_letter("B") == "B"
    assert extract_choice_letter("C) because ...") == "C"
    assert extract_choice_letter("foo bar") is None


def test_extract_letter_restricts_to_choices() -> None:
    assert extract_choice_letter("E is right", choices=("A", "B", "C", "D")) is None
    assert extract_choice_letter("D is the one", choices=("A", "B", "C", "D")) == "D"


def test_mmbench_score_per_category() -> None:
    spec = BenchmarkSpec(name="mmbench")
    ev = MMBenchEvaluator(spec)
    choices = {"A": "cat", "B": "dog", "C": "fish", "D": "bird"}
    exs = [
        Example("1", "", "q?", gold={"letter": "A", "choices": choices, "hint": ""}, category="vis"),
        Example("2", "", "q?", gold={"letter": "B", "choices": choices, "hint": ""}, category="vis"),
        Example("3", "", "q?", gold={"letter": "C", "choices": choices, "hint": ""}, category="ocr"),
    ]
    preds = ["A", "A", "C"]
    out = ev.score(exs, preds)
    assert out["score"] == 2 / 3
    assert out["sub_scores"]["vis"] == 0.5
    assert out["sub_scores"]["ocr"] == 1.0
