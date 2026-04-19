"""Base class + registry for intrinsic metric modules.

Each metric module subclasses :class:`IntrinsicMetric` and registers itself
via :func:`register_metric`. The aggregator in ``experiments.intrinsic.report``
then discovers and dispatches enabled modules from a config list.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class IntrinsicMetric:
    """Contract for all intrinsic metric modules.

    Subclasses set :attr:`name` via :func:`register_metric` and implement
    :meth:`compute`. :meth:`plots` is optional — override when the metric
    has a natural chart representation.
    """

    name: str = ""
    requires_images: bool = False
    requires_cycle_scores: bool = False
    requires_gpu: bool = False

    def compute(self, samples: list[dict], **ctx: Any) -> dict[str, Any]:
        """Return a flat JSON-serialisable dict of this metric's outputs."""
        raise NotImplementedError

    def plots(self, result: dict, out_dir: Path) -> list[Path]:  # noqa: ARG002
        """Optional: write plots to ``out_dir``; return list of files written."""
        return []


METRIC_REGISTRY: dict[str, type[IntrinsicMetric]] = {}


def register_metric(name: str) -> Callable[[type[IntrinsicMetric]], type[IntrinsicMetric]]:
    def wrap(cls: type[IntrinsicMetric]) -> type[IntrinsicMetric]:
        if name in METRIC_REGISTRY:
            raise ValueError(f"metric {name!r} already registered")
        METRIC_REGISTRY[name] = cls
        cls.name = name
        return cls

    return wrap


__all__ = ["IntrinsicMetric", "METRIC_REGISTRY", "register_metric"]
