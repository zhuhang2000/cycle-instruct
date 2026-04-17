"""Unit tests for code/iterative/round_config.py."""
from __future__ import annotations

import pytest

from code.iterative.round_config import (
    RoundTrainingConfig,
    default_schedule,
    get_training_config,
)


def test_round_0_defaults() -> None:
    cfg = get_training_config(0)
    assert cfg.learning_rate == pytest.approx(1e-4)
    assert cfg.num_epochs == 3
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16


def test_round_4_plus_clamped() -> None:
    cfg_4 = get_training_config(4)
    cfg_10 = get_training_config(10)
    assert cfg_4 == cfg_10
    assert cfg_4.learning_rate == pytest.approx(1e-5)
    assert cfg_4.lora_rank == 16


def test_negative_round_clamps_to_zero() -> None:
    assert get_training_config(-5) == get_training_config(0)


def test_lr_monotonic_decrease() -> None:
    lrs = [get_training_config(i).learning_rate for i in range(5)]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1)), lrs


def test_rank_monotonic_increase() -> None:
    ranks = [get_training_config(i).lora_rank for i in range(5)]
    assert all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1)), ranks


def test_epochs_monotonic_decrease() -> None:
    eps = [get_training_config(i).num_epochs for i in range(5)]
    assert all(eps[i] >= eps[i + 1] for i in range(len(eps) - 1)), eps


def test_as_cli_overrides_has_expected_keys() -> None:
    cfg = get_training_config(2)
    flags = cfg.as_cli_overrides()
    for k in ["learning_rate", "num_train_epochs", "lora_rank", "lora_alpha",
              "warmup_ratio", "per_device_train_batch_size",
              "gradient_accumulation_steps", "cutoff_len"]:
        assert k in flags
        assert isinstance(flags[k], str)


def test_custom_schedule() -> None:
    custom = [
        RoundTrainingConfig(1e-3, 5, 4, 8, 0.01),
        RoundTrainingConfig(1e-4, 1, 4, 8, 0.01),
    ]
    assert get_training_config(0, custom).learning_rate == pytest.approx(1e-3)
    assert get_training_config(5, custom).learning_rate == pytest.approx(1e-4)


def test_empty_schedule_raises() -> None:
    with pytest.raises(ValueError):
        get_training_config(0, [])


def test_default_schedule_is_copy() -> None:
    a = default_schedule()
    b = default_schedule()
    a.append(RoundTrainingConfig(0, 0, 0, 0, 0))
    assert len(b) != len(a)
