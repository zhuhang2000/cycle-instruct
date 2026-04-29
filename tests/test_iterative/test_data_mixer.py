"""Unit tests for code/iterative/data_mixer.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from code.iterative.data_mixer import (
    _sample_key,
    deduplicate_by_key,
    mix_training_data,
    sample_from_jsonl,
    to_llamafactory_dataset,
    update_historical_pool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk(qid: int, image: str = "img.jpg", cycle_score: float | None = None) -> dict:
    s = {
        "messages": [
            {"role": "user", "content": f"<image>question_{qid}?"},
            {"role": "assistant", "content": f"answer_{qid}"},
        ],
        "images": [image],
    }
    if cycle_score is not None:
        s["cycle_scores"] = {"composite": cycle_score, "ar": 0.9,
                             "clip": 0.3, "qr": 0.7, "ppl": 0.5}
    return s


def _write_jsonl(path: Path, items: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in items:
            f.write(json.dumps(i, ensure_ascii=False))
            f.write("\n")
    return path


def _write_json_array(path: Path, items: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Keying & dedup
# ---------------------------------------------------------------------------


def test_sample_key_consistent() -> None:
    a, b = _mk(1), _mk(1)
    assert _sample_key(a) == _sample_key(b)


def test_sample_key_differs_on_different_q() -> None:
    assert _sample_key(_mk(1)) != _sample_key(_mk(2))


def test_dedup_keeps_first_occurrence() -> None:
    items = [_mk(1), _mk(1), _mk(2), _mk(2), _mk(1)]
    out = deduplicate_by_key(items)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# sample_from_jsonl
# ---------------------------------------------------------------------------


def test_sample_from_jsonl_missing_returns_empty(tmp_path: Path) -> None:
    assert sample_from_jsonl(tmp_path / "nope.jsonl", 5) == []


def test_sample_from_jsonl_fewer_than_n(tmp_path: Path) -> None:
    items = [_mk(i) for i in range(3)]
    path = _write_jsonl(tmp_path / "p.jsonl", items)
    out = sample_from_jsonl(path, n=5, seed=0)
    assert len(out) == 3


def test_sample_from_jsonl_deterministic(tmp_path: Path) -> None:
    items = [_mk(i) for i in range(100)]
    path = _write_jsonl(tmp_path / "p.jsonl", items)
    a = sample_from_jsonl(path, n=20, seed=42)
    b = sample_from_jsonl(path, n=20, seed=42)
    assert a == b
    c = sample_from_jsonl(path, n=20, seed=7)
    assert a != c


def test_sample_from_jsonl_supports_json_array(tmp_path: Path) -> None:
    items = [_mk(i) for i in range(50)]
    path = _write_json_array(tmp_path / "p.json", items)
    out = sample_from_jsonl(path, n=10, seed=0)
    assert len(out) == 10


# ---------------------------------------------------------------------------
# mix_training_data
# ---------------------------------------------------------------------------


def test_mix_round_0_pure_seed(tmp_path: Path) -> None:
    seed_items = [_mk(i, image=f"seed_{i}.jpg") for i in range(200)]
    seed_path = _write_jsonl(tmp_path / "seed.jsonl", seed_items)
    new = [_mk(i, image=f"new_{i}.jpg") for i in range(500)]

    mixed = mix_training_data(
        new,
        round_id=0,
        target_total=100,
        initial_data_path=seed_path,
        historical_pool_path=None,
        new_ratio_schedule=[0.0, 0.6],
        original_ratio_schedule=[1.0, 0.3],
        historical_ratio_schedule=[0.0, 0.1],
    )

    # All samples must come from the seed set
    seed_keys = {_sample_key(s) for s in seed_items}
    for s in mixed:
        assert _sample_key(s) in seed_keys
    assert len(mixed) == 100


def test_mix_round_1_ratio(tmp_path: Path) -> None:
    seed_items = [_mk(i, image=f"seed_{i}.jpg") for i in range(500)]
    seed_path = _write_jsonl(tmp_path / "seed.jsonl", seed_items)
    hist_items = [_mk(i, image=f"hist_{i}.jpg") for i in range(500)]
    hist_path = _write_jsonl(tmp_path / "hist.jsonl", hist_items)
    new = [_mk(i, image=f"new_{i}.jpg") for i in range(1000)]

    mixed = mix_training_data(
        new,
        round_id=1,
        target_total=1000,
        initial_data_path=seed_path,
        historical_pool_path=hist_path,
        new_ratio_schedule=[0.0, 0.6, 0.5],
        original_ratio_schedule=[1.0, 0.3, 0.3],
        historical_ratio_schedule=[0.0, 0.1, 0.2],
    )

    new_keys = {_sample_key(s) for s in new}
    seed_keys = {_sample_key(s) for s in seed_items}
    hist_keys = {_sample_key(s) for s in hist_items}

    n_new = sum(1 for s in mixed if _sample_key(s) in new_keys)
    n_seed = sum(1 for s in mixed if _sample_key(s) in seed_keys)
    n_hist = sum(1 for s in mixed if _sample_key(s) in hist_keys)

    # Allow ±3% tolerance
    assert 570 <= n_new <= 630
    assert 270 <= n_seed <= 330
    assert 70 <= n_hist <= 130


def test_mix_empty_new_data_does_not_crash(tmp_path: Path) -> None:
    seed_items = [_mk(i) for i in range(100)]
    seed_path = _write_jsonl(tmp_path / "seed.jsonl", seed_items)
    mixed = mix_training_data(
        [],
        round_id=2,
        target_total=50,
        initial_data_path=seed_path,
        historical_pool_path=None,
        new_ratio_schedule=[0.0, 0.6, 0.5],
        original_ratio_schedule=[1.0, 0.3, 0.3],
        historical_ratio_schedule=[0.0, 0.1, 0.2],
    )
    assert len(mixed) > 0  # filled by seed


def test_mix_dedup_across_pools(tmp_path: Path) -> None:
    # Same sample in both seed and new → appears only once in output
    shared = _mk(1)
    seed_path = _write_jsonl(tmp_path / "seed.jsonl", [shared] * 10)
    new = [shared] * 10
    mixed = mix_training_data(
        new,
        round_id=1,
        target_total=10,
        initial_data_path=seed_path,
        historical_pool_path=None,
        new_ratio_schedule=[0.0, 1.0],
        original_ratio_schedule=[1.0, 1.0],
        historical_ratio_schedule=[0.0, 0.0],
    )
    assert len(mixed) == 1


# ---------------------------------------------------------------------------
# historical pool
# ---------------------------------------------------------------------------


def test_historical_pool_quality_filter(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    samples = [
        _mk(0, cycle_score=0.90),  # keep
        _mk(1, cycle_score=0.50),  # drop
        _mk(2, cycle_score=0.88),  # keep
        _mk(3),                    # no score → drop
    ]
    size = update_historical_pool(
        samples, historical_pool_path=pool, pool_size=5000,
        quality_threshold=0.85, round_id=0,
    )
    assert size == 2


def test_historical_pool_cap_keeps_top_k(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    # Add 20 samples, cap at 5, scores 0..19/100
    samples = [_mk(i, cycle_score=i / 100 + 0.80) for i in range(20)]
    size = update_historical_pool(
        samples, historical_pool_path=pool, pool_size=5,
        quality_threshold=0.80, round_id=0,
    )
    assert size == 5
    # verify it kept highest scores
    kept = [json.loads(line) for line in pool.read_text("utf-8").splitlines()]
    scores = sorted([k["cycle_score"] for k in kept], reverse=True)
    assert scores[0] == pytest.approx(0.99)
    assert scores[-1] >= 0.95


def test_historical_pool_repeated_writes_stable(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    for _ in range(3):
        batch = [_mk(i, cycle_score=0.9) for i in range(100)]
        update_historical_pool(
            batch, historical_pool_path=pool, pool_size=50,
            quality_threshold=0.85,
        )
    lines = pool.read_text("utf-8").strip().splitlines()
    assert len(lines) == 50  # capped


def test_historical_pool_upgrades_duplicate_to_higher_score(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    original = _mk(1, cycle_score=0.86)
    upgraded = _mk(1, cycle_score=0.97)

    update_historical_pool(
        [original], historical_pool_path=pool, pool_size=5000,
        quality_threshold=0.85, round_id=0,
    )
    update_historical_pool(
        [upgraded], historical_pool_path=pool, pool_size=5000,
        quality_threshold=0.85, round_id=1,
    )

    kept = [json.loads(line) for line in pool.read_text("utf-8").splitlines()]
    assert len(kept) == 1
    assert kept[0]["cycle_score"] == pytest.approx(0.97)
    assert kept[0]["round_added"] == 1


# ---------------------------------------------------------------------------
# LlamaFactory dataset glue
# ---------------------------------------------------------------------------


def test_to_llamafactory_dataset_writes_both(tmp_path: Path) -> None:
    out_dir = tmp_path / "round_0"
    info_path = tmp_path / "lf_data" / "dataset_info.json"
    samples = [_mk(i) for i in range(10)]
    data_file, info_file = to_llamafactory_dataset(
        samples,
        output_dir=out_dir,
        dataset_name="mixed_round_0",
        dataset_info_path=info_path,
    )
    assert data_file.is_file()
    assert info_file is not None and info_file.is_file()

    info = json.loads(info_file.read_text("utf-8"))
    assert "mixed_round_0" in info
    assert info["mixed_round_0"]["file_name"] == "mixed_round_0.json"
    assert info["mixed_round_0"]["formatting"] == "sharegpt"


def test_to_llamafactory_dataset_preserves_existing_info(tmp_path: Path) -> None:
    info_path = tmp_path / "dataset_info.json"
    info_path.write_text(json.dumps({"existing": {"file_name": "old.json"}}),
                         encoding="utf-8")
    to_llamafactory_dataset(
        [_mk(0)], output_dir=tmp_path, dataset_name="new",
        dataset_info_path=info_path,
    )
    info = json.loads(info_path.read_text("utf-8"))
    assert "existing" in info
    assert "new" in info
