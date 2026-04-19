"""Tests for linguistic_quality: length/template/yes_no/shape."""
from __future__ import annotations

from experiments.intrinsic.linguistic_quality import (
    LinguisticQualityMetric,
    sentence_shape_rate,
    template_repeat_rate,
    yes_no_rate,
)


def test_template_repeat_rate_all_same_prefix():
    texts = ["What is the capital?",
             "What is the answer?",
             "What is the meaning?"]
    assert template_repeat_rate(texts, prefix_tokens=3) == 1.0


def test_template_repeat_rate_all_different():
    texts = ["apples taste sweet",
             "dogs bark loudly",
             "rain falls softly"]
    assert template_repeat_rate(texts, prefix_tokens=3) == 0.0


def test_yes_no_rate():
    answers = ["yes", "no", "Yes.", "Three.", "A cat."]
    assert abs(yes_no_rate(answers) - 0.6) < 1e-9


def test_sentence_shape_rate():
    answers = ["A cat.", "two", "Dog.", "Big!"]
    # 3/4 start with uppercase + end with punctuation
    assert abs(sentence_shape_rate(answers) - 0.75) < 1e-9


def test_linguistic_metric_end_to_end():
    samples = [
        {"question": "What is on the table?", "answer": "A cup."},
        {"question": "What color is the sky?", "answer": "Blue."},
        {"question": "How many dogs?", "answer": "yes"},
    ]
    result = LinguisticQualityMetric().compute(samples)
    assert result["num_samples"] == 3
    assert result["answer_length"]["count"] == 3
    assert 0 <= result["yes_no_answer_rate"] <= 1
