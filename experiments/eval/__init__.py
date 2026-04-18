"""Benchmark evaluation subpackage.

Each benchmark lives in `experiments/eval/benchmarks/<name>.py` and registers
itself via ``@register_benchmark("<name>")``. The runner (`experiments.eval.runner`)
dispatches by name.
"""
from experiments.eval.benchmarks import BENCHMARK_REGISTRY, register_benchmark
from experiments.eval.benchmarks.base import BaseEvaluator, MLLMInferFn

__all__ = [
    "BENCHMARK_REGISTRY",
    "register_benchmark",
    "BaseEvaluator",
    "MLLMInferFn",
]
