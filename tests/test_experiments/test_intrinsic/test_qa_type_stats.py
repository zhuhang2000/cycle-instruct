"""Tests for qa_type_stats: agreement with qa_templates + JS divergence."""
from __future__ import annotations

import math

from experiments.intrinsic.qa_type_stats import (
    QaTypeStats,
    js_divergence,
    per_type_breakdown,
)


def _samples():
    return [
        {"question": "How many cats are in the image?",
         "answer": "Two.", "cycle_scores": {"composite": 0.8}},
        {"question": "How many dogs are there?",
         "answer": "Three.", "cycle_scores": {"composite": 0.4}},
        {"question": "Why is the sky blue?",
         "answer": "Rayleigh scattering.", "cycle_scores": {"composite": 0.9}},
        {"question": "What is to the left of the chair?",
         "answer": "A table.", "cycle_scores": {"composite": 0.6}},
        {"question": "What objects are on the desk?",
         "answer": "A lamp.", "cycle_scores": {"composite": 0.5}},
    ]


def test_compute_matches_qa_templates_distribution():
    from code.iterative.qa_templates import compute_type_distribution
    samples = _samples()
    result = QaTypeStats().compute(samples)
    expected = compute_type_distribution(samples)
    for k in expected:
        assert abs(result["type_distribution"][k] - expected[k]) < 1e-9


def test_js_divergence_symmetric_and_nonneg():
    p = {"a": 0.7, "b": 0.2, "c": 0.1}
    q = {"a": 0.3, "b": 0.4, "c": 0.3}
    d1 = js_divergence(p, q)
    d2 = js_divergence(q, p)
    assert d1 >= 0
    assert math.isclose(d1, d2, abs_tol=1e-9)


def test_js_divergence_identical_is_zero():
    p = {"a": 0.5, "b": 0.5}
    assert js_divergence(p, p) == 0.0


def test_per_type_breakdown_reports_pass_rate():
    samples = _samples()
    bd = per_type_breakdown(samples, composite_threshold=0.7)
    assert "counting" in bd
    # the two counting samples: composite 0.8 and 0.4 -> pass_rate 0.5
    assert abs(bd["counting"]["pass_rate"] - 0.5) < 1e-9
    assert abs(bd["counting"]["mean_composite"] - 0.6) < 1e-9


def test_js_divergence_reported_against_seed():
    samples = _samples()
    seed = [{"question": "What is in the picture?", "answer": "A tree."}] * 5
    result = QaTypeStats().compute(samples, seed_ref=seed)
    assert "js_divergence_vs_seed" in result
    assert result["js_divergence_vs_seed"] >= 0
