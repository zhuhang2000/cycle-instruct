"""Experiments package for Cycle-Instruct.

Subpackages
-----------
eval        : benchmark evaluators (VQAv2 / GQA / MMBench / POPE / DocVQA / HallusionBench)
baselines   : data-generation baselines (no-filter / random / CLIP-only / external datasets)
intrinsic   : data-level quality metrics (diversity / hallucination / cycle-score stats)
analysis    : result aggregators (main table / iteration curves / ablations / efficiency)

Design goals
------------
1. Registry pattern: adding a new benchmark or baseline = drop a file + register.
2. Mock-friendly: all heavy ops (MLLM inference, LoRA training) go through injectable
   hook callables so the scaffolding can be unit-tested on CPU without GPUs.
3. JSON-in / JSON-out: every stage reads and writes plain JSON artifacts to a results
   root, so aggregators can be run independently of the runners.
"""
from __future__ import annotations

from experiments.types import (
    BaselineSpec,
    BenchmarkResult,
    BenchmarkSpec,
    ExperimentSpec,
    MethodRun,
)

__all__ = [
    "BaselineSpec",
    "BenchmarkResult",
    "BenchmarkSpec",
    "ExperimentSpec",
    "MethodRun",
]
