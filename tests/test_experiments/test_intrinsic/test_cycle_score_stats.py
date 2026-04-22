"""Tests for cycle_score_stats: percentiles, correlations, thresholds."""
from __future__ import annotations

from experiments.intrinsic.cycle_score_stats import CycleScoreStats, percentile


def test_percentile_matches_theoretical_on_uniform():
    xs = [i / 100 for i in range(101)]  # 0.00 .. 1.00
    assert abs(percentile(xs, 50) - 0.50) < 1e-2
    assert abs(percentile(xs, 90) - 0.90) < 1e-2


def test_independent_components_near_zero_correlation():
    # Build samples where AR and CLIP are independent (AR increasing,
    # CLIP a repeating pattern).
    samples = [
        {"cycle_scores": {
            "ar": i / 10,
            "clip": [0.2, 0.5, 0.3, 0.8, 0.1, 0.9, 0.4, 0.6, 0.7, 0.5][i],
            "qr": 0.5, "ppl": 0.5, "composite": 0.5,
        }}
        for i in range(10)
    ]
    result = CycleScoreStats().compute(samples)
    corr = result["component_correlations"]["ar"]["clip"]
    # Not required to be 0 exactly — just not strongly correlated
    assert abs(corr) < 0.7


def test_pass_rate_at_threshold_monotonic_decreasing():
    samples = [
        {"cycle_scores": {"ar": 0.5, "clip": 0.3, "qr": 0.5, "ppl": 0.5,
                          "composite": c}}
        for c in [0.4, 0.55, 0.6, 0.7, 0.75, 0.85, 0.9, 0.95]
    ]
    result = CycleScoreStats().compute(
        samples, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    pr = result["pass_rate_at_threshold"]["composite"]
    vals = [pr["0.50"], pr["0.60"], pr["0.70"], pr["0.80"], pr["0.90"]]
    assert vals == sorted(vals, reverse=True)


def test_identity_correlation_is_one():
    samples = [
        {"cycle_scores": {"ar": i / 10, "clip": i / 10, "qr": 0.5,
                          "ppl": 0.5, "composite": 0.5}}
        for i in range(5)
    ]
    result = CycleScoreStats().compute(samples)
    assert abs(result["component_correlations"]["ar"]["ar"] - 1.0) < 1e-9
    assert abs(result["component_correlations"]["ar"]["clip"] - 1.0) < 1e-9
