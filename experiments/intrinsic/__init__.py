"""Data-level (intrinsic) quality metrics for the generated VQA pool.

Importing this package populates :data:`METRIC_REGISTRY` with all built-in
modules. Downstream code resolves module names (``qa_types``, ``diversity``,
``hallucination``, ``cycle_stats``, ``linguistic``, ``alignment``) through
the registry.
"""
from __future__ import annotations

import logging

from experiments.intrinsic.base import (
    IntrinsicMetric,
    METRIC_REGISTRY,
    register_metric,
)

logger = logging.getLogger(__name__)


def _safe_import(module: str) -> None:
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001
        logger.warning("intrinsic module %s failed to import: %s", module, exc)


for _mod in [
    "experiments.intrinsic.qa_type_stats",
    "experiments.intrinsic.diversity",
    "experiments.intrinsic.hallucination",
    "experiments.intrinsic.cycle_score_stats",
    "experiments.intrinsic.linguistic_quality",
    "experiments.intrinsic.alignment",
]:
    _safe_import(_mod)


__all__ = [
    "IntrinsicMetric",
    "METRIC_REGISTRY",
    "register_metric",
]
