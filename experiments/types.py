"""Shared dataclasses used across experiment runners and aggregators.

These types are pure data containers (JSON-serialisable) so that each stage
(prepare-data / train / evaluate / aggregate) can be run in a separate
process and communicate only through artifact files on disk.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ===== Benchmarks =====

@dataclass
class BenchmarkSpec:
    """Declarative description of one downstream benchmark."""

    name: str                        # "vqav2" / "gqa" / "mmbench" / "pope" / ...
    split: str = "val"               # "val" / "test-dev" / "dev" / ...
    data_path: str = ""              # root directory or annotation file
    image_dir: str = ""              # image directory for the benchmark
    max_samples: int | None = None   # cap for smoke tests; None = full split
    metric: str = "accuracy"         # "accuracy" / "anls" / "yes_no_f1" / "mc_acc"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of evaluating one model on one benchmark."""

    benchmark: str
    method: str                      # e.g. "ours_round5" or "baseline.clip_only"
    model_path: str                  # which MLLM was evaluated
    metric: str
    score: float                     # primary headline number
    num_samples: int
    sub_scores: dict[str, float] = field(default_factory=dict)   # e.g. per-category
    predictions_path: str = ""       # where raw predictions JSONL lives
    wall_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ===== Baselines / methods =====

@dataclass
class BaselineSpec:
    """How to produce training data for one baseline method."""

    name: str                        # registry key
    kind: str                        # "filter" | "external_dataset" | "ours"
    params: dict[str, Any] = field(default_factory=dict)
    # For "filter" baselines, these point at the raw generation pool:
    raw_pool_path: str = ""          # JSONL of VQAPair dicts
    # For "external_dataset" baselines, a ready-to-train file:
    dataset_path: str = ""
    # Output
    target_size: int = 100_000
    seed: int = 0


@dataclass
class MethodRun:
    """One complete (prepare -> train -> eval) run of one method."""

    method: str
    baseline: BaselineSpec | None
    training_data_path: str = ""
    trained_model_path: str = ""
    results: list[BenchmarkResult] = field(default_factory=list)
    prepare_time_sec: float = 0.0
    train_time_sec: float = 0.0
    eval_time_sec: float = 0.0
    gpu_hours: float = 0.0


# ===== Full experiment spec =====

@dataclass
class ExperimentSpec:
    """Top-level description of a group of comparable runs.

    All methods in one ExperimentSpec share the same backbone, training
    budget, and benchmark set so their numbers are directly comparable.
    """

    name: str                        # e.g. "main_table_v1"
    backbone: str                    # e.g. "Qwen-VL-2B"
    training_preset: str = "default" # key into training-hyperparam registry
    target_size: int = 100_000       # data budget per method
    methods: list[BaselineSpec] = field(default_factory=list)
    benchmarks: list[BenchmarkSpec] = field(default_factory=list)
    seed: int = 0
    output_root: str = "runs/experiments"


# ===== JSON helpers =====

def _default(obj: Any) -> Any:
    """json.dump default hook: handle Path and dataclasses gracefully."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Type {type(obj)!r} not JSON serialisable")


def save_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_default)


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: Path | str, items: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, default=_default))
            f.write("\n")


def load_jsonl(path: Path | str) -> list[dict]:
    out: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
