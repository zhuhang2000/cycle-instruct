"""Tests for hallucination CHAIRi/CHAIRs with a mock detector."""
from __future__ import annotations

from experiments.intrinsic.hallucination import (
    HallucinationMetric,
    compute_chair,
    extract_noun_phrases,
)


def _samples():
    return [
        {"image_path": "img_0.jpg",
         "answer": "A cat sits on a table near a lamp."},
        {"image_path": "img_1.jpg",
         "answer": "Two dogs are playing in the park."},
        {"image_path": "img_2.jpg",
         "answer": "The skyscraper is blue and tall."},
    ]


def test_all_present_yields_zero_chair():
    samples = _samples()
    # detector says every candidate is present
    chair = compute_chair(samples, lambda img, names: {n: True for n in names},
                         use_spacy=False)
    assert chair["chairi"] == 0.0
    assert chair["chairs"] == 0.0


def test_all_absent_yields_full_chair():
    samples = _samples()
    chair = compute_chair(samples, lambda img, names: {n: False for n in names},
                         use_spacy=False)
    assert chair["chairi"] == 1.0
    assert chair["chairs"] == 1.0


def test_partial_hallucination_per_category_tracked():
    samples = _samples()

    # say "table" is the only noun that's actually present (regardless of image).
    def detector(img, names):
        return {n: (n == "table") for n in names}

    chair = compute_chair(samples, detector, use_spacy=False)
    assert 0.0 < chair["chairi"] < 1.0
    assert "per_category" in chair
    assert chair["total_answers"] == 3
    # table should have total >= 1 and halluc == 0
    assert chair["per_category"]["table"]["halluc"] == 0


def test_hallucination_metric_runs_with_injected_detector():
    samples = _samples()
    metric = HallucinationMetric()
    result = metric.compute(
        samples,
        detector=lambda img, names: {n: True for n in names},
        use_spacy=False,
    )
    assert result["chairi"] == 0.0
    assert result["chairs"] == 0.0


def test_extract_noun_phrases_fallback_is_nonempty():
    nouns = extract_noun_phrases("A red kitten near a lamp.", use_spacy=False)
    # fallback is ≥4-letter alpha tokens — should include kitten, lamp, near
    assert "kitten" in nouns
