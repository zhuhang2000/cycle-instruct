"""Baseline registry.

Each baseline is a ``BaseDataPreparer`` subclass registered under a string
name. Importing this module triggers registration of all built-in baselines.
"""
from __future__ import annotations

from typing import Callable

from experiments.baselines.base import BaseDataPreparer

BASELINE_REGISTRY: dict[str, type[BaseDataPreparer]] = {}


def register_baseline(name: str) -> Callable[[type[BaseDataPreparer]], type[BaseDataPreparer]]:
    def wrap(cls: type[BaseDataPreparer]) -> type[BaseDataPreparer]:
        if name in BASELINE_REGISTRY:
            raise ValueError(f"baseline {name!r} already registered")
        BASELINE_REGISTRY[name] = cls
        cls.baseline_name = name
        return cls

    return wrap


def _safe_import(module: str) -> None:
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "baseline plugin %s failed to import: %s", module, exc,
        )


for _mod in [
    "experiments.baselines.no_filter",
    "experiments.baselines.random_k",
    "experiments.baselines.clip_only",
    "experiments.baselines.length_heuristic",
    "experiments.baselines.external_dataset",
    "experiments.baselines.ours",
]:
    _safe_import(_mod)

__all__ = ["BASELINE_REGISTRY", "register_baseline", "BaseDataPreparer"]
