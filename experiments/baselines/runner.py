"""Baselines runner: prepare → train → evaluate → write MethodRun.

Given an ExperimentSpec YAML, iterate over methods, invoke each baseline's
``prepare`` to produce a training JSON, invoke the training hook to obtain
a trained model path, then invoke ``experiments.eval.runner.run_benchmarks``
and persist a ``MethodRun`` JSON alongside the per-benchmark results.

Training and inference are both injectable hooks so the entire orchestration
can be exercised on CPU (no real GPU call) in unit tests.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from experiments.baselines import BASELINE_REGISTRY
from experiments.eval.benchmarks import BENCHMARK_REGISTRY  # noqa: F401  (trigger)
from experiments.eval.benchmarks.base import MLLMInferFn
from experiments.eval.runner import (
    _stub_infer_fn,
    default_infer_fn,
    load_benchmarks_from_spec,
    run_benchmarks,
)
from experiments.types import (
    BaselineSpec,
    ExperimentSpec,
    MethodRun,
    save_json,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training hook
# ---------------------------------------------------------------------------
TrainFn = Callable[[Path, Path], Path]   # (dataset_json, output_dir) -> model_path


def _stub_train_fn() -> TrainFn:
    """Creates a fake model directory with a sentinel file."""

    def _fn(dataset: Path, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.stub").write_text(f"stub-model-from:{dataset}\n")
        return out_dir

    return _fn


def default_train_fn(backbone: str, preset: str = "default") -> TrainFn:
    """Real training via the LlamaFactory / bash pipeline — subprocess call.

    Writes a temporary shell invocation to ``out_dir/train.log``. Raises
    RuntimeError on non-zero exit. Kept thin on purpose so tests don't need
    to patch LlamaFactory internals.
    """
    import subprocess  # noqa: WPS433

    def _fn(dataset: Path, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).resolve().parents[2] / "bash" / "run_multimodal_cycle.sh"
        if not script.exists():
            raise FileNotFoundError(f"training script not found: {script}")
        cmd = [
            "bash", str(script),
            "--dataset", str(dataset),
            "--backbone", backbone,
            "--preset", preset,
            "--output", str(out_dir),
        ]
        log_path = out_dir / "train.log"
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"training failed (exit {proc.returncode}); see {log_path}")
        return out_dir

    return _fn


# ---------------------------------------------------------------------------
# Single-method orchestration
# ---------------------------------------------------------------------------

def run_method(
    spec: BaselineSpec,
    experiment: ExperimentSpec,
    train_fn: TrainFn,
    infer_fn: MLLMInferFn,
    output_root: Path,
) -> MethodRun:
    cls = BASELINE_REGISTRY.get(spec.kind if spec.kind in BASELINE_REGISTRY else spec.name)
    if cls is None:
        # ``kind`` is the conceptual family (filter/external_dataset/ours);
        # for filter baselines the actual baseline is ``name``.
        cls = BASELINE_REGISTRY.get(spec.name)
    if cls is None:
        raise ValueError(f"baseline {spec.name!r} (kind={spec.kind!r}) not registered")

    method_dir = Path(output_root) / spec.name
    method_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    preparer = cls(spec)
    dataset_path = preparer.prepare(method_dir / "data")
    prep_dt = time.time() - t0

    t1 = time.time()
    model_path = train_fn(dataset_path, method_dir / "model")
    train_dt = time.time() - t1

    t2 = time.time()
    results = run_benchmarks(
        benchmarks=experiment.benchmarks,
        infer_fn=infer_fn,
        model_path=str(model_path),
        method_name=spec.name,
        output_dir=method_dir,
    )
    eval_dt = time.time() - t2

    run = MethodRun(
        method=spec.name,
        baseline=spec,
        training_data_path=str(dataset_path),
        trained_model_path=str(model_path),
        results=results,
        prepare_time_sec=prep_dt,
        train_time_sec=train_dt,
        eval_time_sec=eval_dt,
    )
    save_json(method_dir / "run.json", asdict(run))
    logger.info("[baselines] %s done in %.1fs (prep %.1f / train %.1f / eval %.1f)",
                spec.name, prep_dt + train_dt + eval_dt, prep_dt, train_dt, eval_dt)
    return run


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

def load_experiment_spec(path: Path, smoke: bool = False) -> ExperimentSpec:
    import yaml  # type: ignore
    raw = yaml.safe_load(Path(path).read_text("utf-8"))
    benches = load_benchmarks_from_spec(path, smoke=smoke)
    methods = [
        BaselineSpec(
            name=m["name"],
            kind=m.get("kind", "filter"),
            params=m.get("params", {}),
            raw_pool_path=m.get("raw_pool_path", ""),
            dataset_path=m.get("dataset_path", ""),
            target_size=m.get("target_size", raw.get("target_size", 100_000)),
            seed=m.get("seed", raw.get("seed", 0)),
        )
        for m in raw.get("methods", [])
    ]
    return ExperimentSpec(
        name=raw["name"],
        backbone=raw.get("backbone", ""),
        training_preset=raw.get("training_preset", "default"),
        target_size=raw.get("target_size", 100_000),
        methods=methods,
        benchmarks=benches,
        seed=raw.get("seed", 0),
        output_root=raw.get("output_root", "runs/experiments"),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Run all baselines × benchmarks.")
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--only", type=str, default="", help="comma-sep method names filter")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-root", type=Path, default=None)
    args = ap.parse_args(argv)

    experiment = load_experiment_spec(args.spec, smoke=args.smoke)
    out_root = args.output_root or Path(experiment.output_root) / experiment.name

    if args.smoke or os.environ.get("CI_SKIP_HEAVY") == "1":
        train_fn = _stub_train_fn()
        infer_fn: MLLMInferFn = _stub_infer_fn()
    else:
        train_fn = default_train_fn(experiment.backbone, experiment.training_preset)
        infer_fn = default_infer_fn()

    only = set(s.strip() for s in args.only.split(",") if s.strip())
    runs: list[MethodRun] = []
    for m in experiment.methods:
        if only and m.name not in only:
            continue
        try:
            runs.append(run_method(m, experiment, train_fn, infer_fn, out_root))
        except Exception as exc:  # noqa: BLE001
            logger.exception("method %s failed: %s", m.name, exc)

    save_json(out_root / "experiment.json", {
        "experiment": experiment.name,
        "num_methods": len(runs),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
