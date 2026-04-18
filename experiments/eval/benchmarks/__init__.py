"""Benchmark registry.

Importing this module loads every benchmark plugin. Each plugin calls
``register_benchmark`` at import time, populating ``BENCHMARK_REGISTRY``.
"""
from __future__ import annotations

from typing import Callable

from experiments.eval.benchmarks.base import BaseEvaluator

BENCHMARK_REGISTRY: dict[str, type[BaseEvaluator]] = {}


def register_benchmark(name: str) -> Callable[[type[BaseEvaluator]], type[BaseEvaluator]]:
    """Decorator to register a concrete evaluator class under a string name."""

    def wrap(cls: type[BaseEvaluator]) -> type[BaseEvaluator]:
        if name in BENCHMARK_REGISTRY:
            raise ValueError(f"benchmark {name!r} already registered")
        BENCHMARK_REGISTRY[name] = cls
        cls.benchmark_name = name
        return cls

    return wrap


# Trigger registration by importing concrete modules.
# Guarded so unit tests that import just the registry don't fail if a
# benchmark module has a heavy optional dependency.
def _safe_import(module: str) -> None:
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001 - intentional broad guard
        import logging
        logging.getLogger(__name__).warning(
            "benchmark plugin %s failed to import: %s", module, exc,
        )


for _mod in [
    "experiments.eval.benchmarks.vqav2",
    "experiments.eval.benchmarks.gqa",
    "experiments.eval.benchmarks.mmbench",
    "experiments.eval.benchmarks.pope",
    "experiments.eval.benchmarks.docvqa",
    "experiments.eval.benchmarks.hallusion",
]:
    _safe_import(_mod)

__all__ = ["BENCHMARK_REGISTRY", "register_benchmark"]
