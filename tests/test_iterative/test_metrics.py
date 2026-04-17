"""Unit tests for code/iterative/metrics.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.iterative.metrics import (
    RoundMetrics,
    load_all_rounds,
    save_metrics,
    should_stop,
)


def _make(round_id: int, **kw) -> RoundMetrics:
    return RoundMetrics(round_id=round_id, **kw)


def test_metrics_roundtrip(tmp_path: Path) -> None:
    m = RoundMetrics(
        round_id=2,
        num_samples_generated=100,
        num_samples_passed_filter=60,
        pass_rate=0.6,
        mean_cycle_score=0.75,
        std_cycle_score=0.1,
        mean_ar=0.8, mean_clip=0.3, mean_qr=0.7, mean_ppl=0.5,
        train_loss_initial=2.0, train_loss_final=0.9,
        lora_rank=16, learning_rate=5e-5, num_epochs=2,
        data_diversity_score=0.85,
        drift_from_prev=0.07,
    )
    save_metrics(tmp_path / "round_2", m)

    (tmp_path / "round_2" / "metrics.json").is_file()
    raw = json.loads((tmp_path / "round_2" / "metrics.json").read_text("utf-8"))
    reloaded = RoundMetrics.from_dict(raw)

    assert reloaded.round_id == 2
    assert reloaded.pass_rate == pytest.approx(0.6)
    assert reloaded.data_diversity_score == pytest.approx(0.85)


def test_load_all_rounds_sorted(tmp_path: Path) -> None:
    for rid in [2, 0, 1]:
        save_metrics(tmp_path / f"round_{rid}", _make(rid))
    loaded = load_all_rounds(tmp_path)
    assert [m.round_id for m in loaded] == [0, 1, 2]


def test_load_all_rounds_missing_dir(tmp_path: Path) -> None:
    assert load_all_rounds(tmp_path / "does_not_exist") == []


def test_should_stop_empty_history() -> None:
    stop, reason = should_stop([])
    assert stop is False
    assert "empty" in reason


def test_should_stop_max_rounds() -> None:
    # 5 rounds of history, max_rounds=5 → stop
    history = [_make(i) for i in range(5)]
    stop, reason = should_stop(history, max_rounds=5)
    assert stop is True
    assert "max-rounds" in reason


def test_should_stop_pass_rate_collapse() -> None:
    history = [
        _make(0, pass_rate=0.80),
        _make(1, pass_rate=0.60),  # drop 0.20 > 0.15
        _make(2, pass_rate=0.40),  # drop 0.20 > 0.15
    ]
    stop, reason = should_stop(history, pass_rate_drop_threshold=0.15, patience=2)
    assert stop is True
    assert "pass-rate" in reason


def test_should_stop_cycle_score_decreasing() -> None:
    history = [
        _make(0, pass_rate=0.8, mean_cycle_score=0.80),
        _make(1, pass_rate=0.8, mean_cycle_score=0.75),
        _make(2, pass_rate=0.8, mean_cycle_score=0.70),
    ]
    stop, reason = should_stop(history, patience=2)
    assert stop is True
    assert "cycle-score" in reason


def test_should_stop_diversity_collapse() -> None:
    history = [
        _make(0, data_diversity_score=0.9),
        _make(1, data_diversity_score=0.55),  # below 0.6
    ]
    stop, reason = should_stop(history, diversity_threshold=0.6)
    assert stop is True
    assert "diversity-collapse" in reason


def test_should_stop_diversity_zero() -> None:
    history = [
        _make(0, data_diversity_score=0.0),
    ]
    stop, reason = should_stop(history, diversity_threshold=0.6)
    assert stop is True
    assert "diversity-collapse" in reason


def test_should_stop_drift_converged() -> None:
    history = [
        _make(0, drift_from_prev=None),
        _make(1, drift_from_prev=0.01),  # below 0.02
    ]
    stop, reason = should_stop(history, drift_converged_threshold=0.02)
    assert stop is True
    assert "drift" in reason


def test_should_stop_healthy() -> None:
    history = [
        _make(0, pass_rate=0.70, mean_cycle_score=0.75, data_diversity_score=0.85),
        _make(1, pass_rate=0.72, mean_cycle_score=0.76, data_diversity_score=0.85,
              drift_from_prev=0.10),
        _make(2, pass_rate=0.71, mean_cycle_score=0.77, data_diversity_score=0.86,
              drift_from_prev=0.08),
    ]
    stop, reason = should_stop(history, patience=2)
    assert stop is False
    assert reason == "healthy"


def test_should_stop_insufficient_history() -> None:
    # Not enough rounds to evaluate trend rules — rules 3/5 still apply though
    history = [_make(0, pass_rate=0.8, data_diversity_score=0.85)]
    stop, reason = should_stop(history, patience=2)
    assert stop is False
    assert reason == "insufficient-history"


def test_from_dict_unknown_keys_go_to_extras() -> None:
    raw = {"round_id": 1, "mystery_metric": 0.42}
    m = RoundMetrics.from_dict(raw)
    assert m.round_id == 1
    assert m.extras.get("mystery_metric") == 0.42
