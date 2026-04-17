"""End-to-end smoke test of the iterative controller with mocked hooks.

The three heavy hooks (`gen_filter_fn`, `train_fn`, `merge_fn`) are
replaced with fakes so the test can run on CPU in < 1 s and verify:

* the correct number of rounds execute,
* round directories + metrics files are created,
* the mixing pipeline actually produces a ShareGPT dataset file,
* the historical pool accumulates high-quality samples across rounds,
* the second round's generator is the first round's ``merged_model``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from code.iterative.iterative_trainer import (
    IterativeConfig,
    run_iterative_training,
)
from code.iterative.round_config import RoundTrainingConfig


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _fake_vqa(idx: int, *, score: float) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"<image>question_{idx}?"},
            {"role": "assistant", "content": f"answer_{idx}"},
        ],
        "images": [f"img_{idx}.jpg"],
        "cycle_scores": {
            "composite": score,
            "ar": score,
            "clip": 0.30,
            "qr": 0.65,
            "ppl": 0.50,
        },
    }


class _FakeHooks:
    """Bundle of fake hooks that track calls for assertions."""

    def __init__(self) -> None:
        self.gen_calls: list[tuple[str, int]] = []
        self.train_calls: list[tuple[str, Path, int]] = []
        self.merge_calls: list[tuple[str, Path, Path]] = []

    def gen_filter(
        self,
        generator_model_path: str,
        raw_image_dir: str,
        round_dir: Path,
        samples_to_generate: int,
    ) -> list[dict[str, Any]]:
        self.gen_calls.append((generator_model_path, samples_to_generate))
        # Produce a mix of high-score + low-score VQAs to exercise the
        # historical-pool quality filter (threshold default 0.85).
        out = []
        # 30 high-score samples (pass 0.85 threshold)
        for i in range(30):
            out.append(_fake_vqa(i + samples_to_generate * 100, score=0.90))
        # 20 low-score samples (dropped from historical pool)
        for i in range(20):
            out.append(_fake_vqa(i + samples_to_generate * 200, score=0.50))
        # Write a fake raw_vqa.json so num_generated reads correctly
        raw = round_dir / "raw_vqa.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        with raw.open("w", encoding="utf-8") as f:
            json.dump(out + [_fake_vqa(999 + i, score=0.2) for i in range(50)], f)
        return out

    def train(
        self,
        base_model_path: str,
        dataset_file: Path,
        train_cfg: RoundTrainingConfig,
        round_dir: Path,
        dataset_name: str,
    ) -> Path:
        self.train_calls.append((base_model_path, dataset_file, train_cfg.lora_rank))
        lora_dir = round_dir / "lora"
        lora_dir.mkdir(parents=True, exist_ok=True)
        (lora_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        # Fake a trainer_log.jsonl so loss-reading doesn't crash
        (lora_dir / "trainer_log.jsonl").write_text(
            '{"loss": 2.1}\n{"loss": 0.9}\n', encoding="utf-8",
        )
        return lora_dir

    def merge(
        self,
        base_model_path: str,
        lora_dir: Path,
        merged_dir: Path,
    ) -> None:
        self.merge_calls.append((base_model_path, lora_dir, merged_dir))
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "config.json").write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def iterative_cfg(tmp_path: Path) -> IterativeConfig:
    base_model = tmp_path / "base_model"
    base_model.mkdir()
    (base_model / "config.json").write_text("{}", encoding="utf-8")

    seed_file = tmp_path / "seed.jsonl"
    seed_records = [
        {
            "messages": [
                {"role": "user", "content": f"<image>seed_q_{i}?"},
                {"role": "assistant", "content": f"seed_a_{i}"},
            ],
            "images": [f"seed_{i}.jpg"],
        }
        for i in range(100)
    ]
    with seed_file.open("w", encoding="utf-8") as f:
        for r in seed_records:
            f.write(json.dumps(r))
            f.write("\n")

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # not used by fake gen, but must exist
    (images_dir / "img_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    output_root = tmp_path / "run"

    return IterativeConfig(
        base_model_path=str(base_model),
        initial_data_path=str(seed_file),
        raw_image_dir=str(images_dir),
        output_root=str(output_root),
        max_rounds=2,
        samples_per_round=30,
        new_ratio_schedule=[0.0, 0.6],
        original_ratio_schedule=[1.0, 0.3],
        historical_ratio_schedule=[0.0, 0.1],
        historical_pool_size=100,
        historical_quality_threshold=0.85,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_two_round_integration(iterative_cfg: IterativeConfig) -> None:
    hooks = _FakeHooks()
    history = run_iterative_training(
        iterative_cfg,
        gen_filter_fn=hooks.gen_filter,
        train_fn=hooks.train,
        merge_fn=hooks.merge,
    )

    # Two rounds ran, barring early-stop
    assert len(history) == 2
    assert [m.round_id for m in history] == [0, 1]

    # Directory structure
    round0 = Path(iterative_cfg.output_root) / "round_0"
    round1 = Path(iterative_cfg.output_root) / "round_1"
    assert (round0 / "metrics.json").is_file()
    assert (round1 / "metrics.json").is_file()
    assert (round0 / "mixed_round_0.json").is_file()
    assert (round1 / "mixed_round_1.json").is_file()
    assert (round0 / "lora" / "adapter_config.json").is_file()
    assert (round0 / "merged_model" / "config.json").is_file()

    # Historical pool accumulates across rounds
    pool = Path(iterative_cfg.output_root) / "historical_pool.jsonl"
    assert pool.is_file()
    pool_lines = pool.read_text("utf-8").strip().splitlines()
    assert len(pool_lines) > 0
    # Pool only holds high-quality samples (>= threshold 0.85)
    for line in pool_lines:
        rec = json.loads(line)
        assert rec["cycle_score"] >= 0.85


def test_generator_rotates_to_prev_merged(iterative_cfg: IterativeConfig) -> None:
    hooks = _FakeHooks()
    run_iterative_training(
        iterative_cfg,
        gen_filter_fn=hooks.gen_filter,
        train_fn=hooks.train,
        merge_fn=hooks.merge,
    )
    # Round 0 uses the base model; Round 1 uses round_0's merged_model
    assert hooks.gen_calls[0][0] == iterative_cfg.base_model_path
    expected_round1 = str(Path(iterative_cfg.output_root) / "round_0" / "merged_model")
    assert hooks.gen_calls[1][0] == expected_round1


def test_lora_always_from_base_model(iterative_cfg: IterativeConfig) -> None:
    """Regression: every round's LoRA training starts from base_model_path."""
    hooks = _FakeHooks()
    run_iterative_training(
        iterative_cfg,
        gen_filter_fn=hooks.gen_filter,
        train_fn=hooks.train,
        merge_fn=hooks.merge,
    )
    for call in hooks.train_calls:
        assert call[0] == iterative_cfg.base_model_path


def test_metrics_contain_expected_fields(iterative_cfg: IterativeConfig) -> None:
    hooks = _FakeHooks()
    history = run_iterative_training(
        iterative_cfg,
        gen_filter_fn=hooks.gen_filter,
        train_fn=hooks.train,
        merge_fn=hooks.merge,
    )
    m = history[-1]
    assert 0.0 <= m.pass_rate <= 1.0
    assert m.lora_rank > 0
    assert m.learning_rate > 0
    assert m.mean_cycle_score > 0
    # Round 1 should have a non-None drift (previous round exists)
    assert m.drift_from_prev is not None


def test_early_stop_on_diversity_collapse(tmp_path: Path) -> None:
    """If the fake generator returns only one QA type, diversity should trip."""
    base_model = tmp_path / "base_model"
    base_model.mkdir()

    seed_file = tmp_path / "seed.jsonl"
    # Seed only with "counting" questions → diversity should be very low
    with seed_file.open("w", encoding="utf-8") as f:
        for i in range(50):
            rec = {
                "messages": [
                    {"role": "user", "content": f"<image>How many X_{i} are there?"},
                    {"role": "assistant", "content": str(i)},
                ],
                "images": [f"i_{i}.jpg"],
            }
            f.write(json.dumps(rec))
            f.write("\n")

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "i.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    cfg = IterativeConfig(
        base_model_path=str(base_model),
        initial_data_path=str(seed_file),
        raw_image_dir=str(images_dir),
        output_root=str(tmp_path / "run"),
        max_rounds=3,
        samples_per_round=20,
        new_ratio_schedule=[0.0, 0.6, 0.5],
        original_ratio_schedule=[1.0, 0.3, 0.3],
        historical_ratio_schedule=[0.0, 0.1, 0.2],
        diversity_threshold=0.60,
    )

    hooks = _FakeHooks()
    history = run_iterative_training(
        cfg,
        gen_filter_fn=hooks.gen_filter,
        train_fn=hooks.train,
        merge_fn=hooks.merge,
    )
    # Round 0 ran and its diversity < 0.6 → should_stop should have tripped
    assert len(history) == 1
    assert history[0].data_diversity_score is not None
    assert history[0].data_diversity_score < 0.60
