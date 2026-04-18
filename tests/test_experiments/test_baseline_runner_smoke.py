"""End-to-end smoke: mock infer + mock train, 2 baselines × 1 benchmark.

Validates directory layout and that run.json carries the expected fields.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.baselines.runner import _stub_train_fn, run_method
from experiments.eval.runner import _stub_infer_fn
from experiments.types import BaselineSpec, BenchmarkSpec, ExperimentSpec


def _write_pool(path: Path, n: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            rec = {
                "image_path": f"img_{i}.jpg",
                "question": f"What object {i}?",
                "answer": f"answer_{i}",
                "cycle_scores": {"clip": 0.3 + 0.02 * i, "composite": 0.8},
            }
            f.write(json.dumps(rec) + "\n")


def _pope_file(path: Path, n: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "question_id": i,
                "image": f"img_{i}.jpg",
                "text": "is there a cat?",
                "label": "yes" if i % 2 == 0 else "no",
                "category": "random",
            }) + "\n")


@pytest.mark.parametrize("baseline_name,kind,params", [
    ("no_filter", "no_filter", {}),
    ("clip_only", "clip_only", {"clip_threshold": 0.3}),
])
def test_single_method_smoke(tmp_path: Path, baseline_name: str, kind: str, params: dict) -> None:
    pool = tmp_path / "pool.jsonl"
    _write_pool(pool)
    pope_path = tmp_path / "pope.jsonl"
    _pope_file(pope_path)
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    bench = BenchmarkSpec(
        name="pope",
        data_path=str(pope_path),
        image_dir=str(img_dir),
        max_samples=3,
    )
    spec = BaselineSpec(
        name=baseline_name,
        kind=kind,
        params=params,
        raw_pool_path=str(pool),
        target_size=100,
    )
    experiment = ExperimentSpec(
        name="smoke",
        backbone="stub",
        methods=[spec],
        benchmarks=[bench],
    )

    out_root = tmp_path / "runs"
    run = run_method(
        spec=spec,
        experiment=experiment,
        train_fn=_stub_train_fn(),
        infer_fn=_stub_infer_fn("yes"),
        output_root=out_root,
    )

    assert run.method == baseline_name
    assert Path(run.training_data_path).exists()
    assert Path(run.trained_model_path).exists()
    assert len(run.results) == 1
    assert run.results[0].benchmark == "pope"
    assert (out_root / baseline_name / "run.json").exists()
