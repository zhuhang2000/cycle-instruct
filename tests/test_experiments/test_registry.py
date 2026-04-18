"""Registry: every built-in benchmark / baseline is discoverable & no dupes."""
from __future__ import annotations

import pytest

from experiments.baselines import BASELINE_REGISTRY, register_baseline
from experiments.baselines.base import BaseDataPreparer
from experiments.eval.benchmarks import BENCHMARK_REGISTRY, register_benchmark
from experiments.eval.benchmarks.base import BaseEvaluator


def test_benchmark_registry_has_all_six() -> None:
    for name in ("vqav2", "gqa", "mmbench", "pope", "docvqa", "hallusion"):
        assert name in BENCHMARK_REGISTRY, f"{name} missing from benchmark registry"
        cls = BENCHMARK_REGISTRY[name]
        assert issubclass(cls, BaseEvaluator)
        assert cls.benchmark_name == name


def test_baseline_registry_has_all_six() -> None:
    for name in ("no_filter", "random_k", "clip_only",
                 "length_heuristic", "external_dataset", "ours"):
        assert name in BASELINE_REGISTRY, f"{name} missing from baseline registry"
        assert issubclass(BASELINE_REGISTRY[name], BaseDataPreparer)


def test_duplicate_benchmark_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        @register_benchmark("vqav2")
        class _Dup(BaseEvaluator):
            pass


def test_duplicate_baseline_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):
        @register_baseline("no_filter")
        class _Dup(BaseDataPreparer):
            pass
