"""Aggregator golden-ish tests: small synthetic run tree → expected table cells."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.analysis.aggregate_tables import aggregate, build_grid, to_markdown
from experiments.analysis.ablation_table import aggregate_ablation


def _write_run(dir: Path, method: str, results: list[dict]) -> None:
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "run.json").write_text(json.dumps({
        "method": method,
        "results": results,
    }))


def test_aggregate_main_table(tmp_path: Path) -> None:
    exp = tmp_path / "main_v1"
    _write_run(exp / "a", "a", [
        {"benchmark": "vqav2", "score": 0.5, "metric": "accuracy", "num_samples": 100},
        {"benchmark": "pope", "score": 0.8, "metric": "accuracy", "num_samples": 100},
    ])
    _write_run(exp / "b", "b", [
        {"benchmark": "vqav2", "score": 0.6, "metric": "accuracy", "num_samples": 100},
        # pope missing on purpose
    ])

    paths = aggregate(exp, exp / "tables")
    md = Path(paths["md"]).read_text()
    assert "| a |" in md
    assert "| b |" in md
    assert "0.500" in md and "0.800" in md
    assert "--" in md  # missing cell


def test_build_grid_stable_ordering(tmp_path: Path) -> None:
    exp = tmp_path / "x"
    _write_run(exp / "m1", "m1", [
        {"benchmark": "vqav2", "score": 0.1},
        {"benchmark": "pope",  "score": 0.2},
    ])
    _write_run(exp / "m2", "m2", [
        {"benchmark": "mmbench", "score": 0.3},
    ])
    from experiments.analysis.aggregate_tables import collect_runs
    methods, benches, cells = build_grid(collect_runs(exp))
    assert methods == ["m1", "m2"]
    assert set(benches) == {"vqav2", "pope", "mmbench"}
    assert cells[("m1", "vqav2")]["score"] == 0.1


def test_ablation_delta_format(tmp_path: Path) -> None:
    exp = tmp_path / "abl"
    _write_run(exp / "full", "full", [{"benchmark": "pope", "score": 0.80}])
    _write_run(exp / "drop_ar", "drop_ar", [{"benchmark": "pope", "score": 0.70}])

    paths = aggregate_ablation(exp, exp / "tables", full_method="full")
    md = Path(paths["md"]).read_text()
    assert "0.700" in md
    assert "-0.100" in md  # delta
