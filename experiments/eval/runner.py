"""Evaluation runner: one method × N benchmarks.

Given a spec YAML (or a single benchmark name + data paths on the CLI),
iterate through ``BENCHMARK_REGISTRY`` and produce one ``BenchmarkResult``
JSON per benchmark. Inference is delegated to an injectable hook so unit
tests and CI can run without a GPU.

CLI
---
    python -m experiments.eval.runner \\
        --spec experiments/configs/main_table.yaml \\
        --method ours_round5 \\
        --model-path /path/to/merged_model \\
        --output-dir runs/experiments/main_table_v1/ours_round5

    # Smoke: stub inference, all benchmarks capped at 20 samples
    python -m experiments.eval.runner --spec ... --method ... --smoke
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from experiments.eval.benchmarks import BENCHMARK_REGISTRY
from experiments.eval.benchmarks.base import MLLMInferFn
from experiments.types import BenchmarkResult, BenchmarkSpec, save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference hook factory
# ---------------------------------------------------------------------------

def _stub_infer_fn(stub_text: str = "yes") -> MLLMInferFn:
    """CPU-only stub — returns a constant string. Used by --smoke and tests."""

    def _fn(messages: list[dict], images: list[str], model_path: str) -> str:
        return stub_text

    return _fn


def default_infer_fn() -> MLLMInferFn:
    """Real inference via ``tool.multimodal_infer.generate_multimodal``.

    Imported lazily so that unit tests (which don't have torch/vllm) can still
    import this module.
    """
    from tool.multimodal_infer import generate_multimodal  # noqa: WPS433
    from tool.multimodal_types import ImageTextSample, MultimodalInferConfig

    cfg_cache: dict[str, Any] = {}

    def _fn(messages: list[dict], images: list[str], model_path: str) -> str:
        cfg = cfg_cache.get(model_path)
        if cfg is None:
            cfg = MultimodalInferConfig(mllm_model_path=model_path)
            cfg_cache[model_path] = cfg
        sample = ImageTextSample(
            image_path=images[0] if images else "",
            image_id="eval",
            source_text=None,
            source_type="eval",
            metadata={},
        )

        def _build(_: ImageTextSample) -> tuple[list[dict], list[str]]:
            return messages, images

        def _rec(_: ImageTextSample, raw: str) -> dict:
            return {"raw": raw}

        # One-shot call: generate_multimodal is batch-oriented, but for the
        # evaluator we want a single string. Callers that want throughput
        # should override this hook with a batched version.
        out = generate_multimodal([sample], _build, Path("/dev/null"), _rec, cfg)
        return out[0].get("raw", "") if out else ""

    return _fn


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------

def run_benchmarks(
    benchmarks: list[BenchmarkSpec],
    infer_fn: MLLMInferFn,
    model_path: str,
    method_name: str,
    output_dir: Path,
) -> list[BenchmarkResult]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []
    for spec in benchmarks:
        cls = BENCHMARK_REGISTRY.get(spec.name)
        if cls is None:
            logger.warning("benchmark %r not registered; skipping", spec.name)
            continue
        evaluator = cls(spec)
        bench_out = output_dir / "results"
        logger.info("[eval] %s :: %s", method_name, spec.name)
        result = evaluator.evaluate(infer_fn, model_path, method_name, bench_out)
        save_json(bench_out / f"{spec.name}.json", result.to_dict())
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyyaml is required for --spec") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_benchmarks_from_spec(
    spec_path: Path,
    smoke: bool = False,
) -> list[BenchmarkSpec]:
    raw = _load_yaml(spec_path)
    benches: list[BenchmarkSpec] = []
    for item in raw.get("benchmarks", []):
        params = dict(item)
        if smoke:
            params["max_samples"] = min(params.get("max_samples") or 20, 20)
        extras = params.pop("extras", {})
        spec = BenchmarkSpec(
            name=params["name"],
            split=params.get("split", "val"),
            data_path=params.get("data_path", ""),
            image_dir=params.get("image_dir", ""),
            max_samples=params.get("max_samples"),
            metric=params.get("metric", "accuracy"),
            extras=extras,
        )
        benches.append(spec)
    return benches


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one method across benchmarks.")
    p.add_argument("--spec", type=Path, help="experiment YAML (benchmarks section read)")
    p.add_argument("--method", type=str, default="unnamed_method")
    p.add_argument("--model-path", type=str, default="")
    p.add_argument("--output-dir", type=Path, default=Path("runs/eval_adhoc"))
    p.add_argument("--benchmark", type=str, help="single benchmark name (no spec)")
    p.add_argument("--data-path", type=str, default="")
    p.add_argument("--image-dir", type=str, default="")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="use stub infer, cap samples")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_argparser().parse_args(argv)

    if args.spec:
        benches = load_benchmarks_from_spec(args.spec, smoke=args.smoke)
    elif args.benchmark:
        benches = [BenchmarkSpec(
            name=args.benchmark,
            split=args.split,
            data_path=args.data_path,
            image_dir=args.image_dir,
            max_samples=args.max_samples if not args.smoke else (args.max_samples or 20),
        )]
    else:
        raise SystemExit("must supply either --spec or --benchmark")

    if args.smoke or os.environ.get("CI_SKIP_HEAVY") == "1":
        infer_fn: MLLMInferFn = _stub_infer_fn()
    else:
        infer_fn = default_infer_fn()

    results = run_benchmarks(
        benchmarks=benches,
        infer_fn=infer_fn,
        model_path=args.model_path,
        method_name=args.method,
        output_dir=args.output_dir,
    )
    save_json(Path(args.output_dir) / "run.json", {
        "method": args.method,
        "model_path": args.model_path,
        "results": [asdict(r) for r in results],
    })
    logger.info("[eval] wrote %d benchmark result(s) to %s",
                len(results), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
