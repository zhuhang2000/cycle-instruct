"""Cycle-score descriptive stats over a VQA pool.

For each score component (ar/clip/qr/ppl/composite) reports mean, std,
percentiles, and the pass-rate at common thresholds.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Iterable


_COMPONENTS = ("ar", "clip", "qr", "ppl", "composite")


def _collect(samples: Iterable[dict], comp: str) -> list[float]:
    out = []
    for s in samples:
        cs = s.get("cycle_scores") or {}
        v = cs.get(comp)
        if v is not None:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    vs = sorted(values)
    k = max(0, min(len(vs) - 1, int(round((q / 100) * (len(vs) - 1)))))
    return vs[k]


def summarise(samples: list[dict],
              thresholds: dict[str, float] | None = None) -> dict:
    thresholds = thresholds or {
        "composite": 0.7, "ar": 0.6, "clip": 0.2, "qr": 0.55,
    }
    out: dict = {"num_samples": len(samples), "components": {}}
    for comp in _COMPONENTS:
        vs = _collect(samples, comp)
        if not vs:
            out["components"][comp] = {"count": 0}
            continue
        out["components"][comp] = {
            "count": len(vs),
            "mean": mean(vs),
            "std": pstdev(vs) if len(vs) > 1 else 0.0,
            "p10": _percentile(vs, 10),
            "p50": _percentile(vs, 50),
            "p90": _percentile(vs, 90),
            "min": min(vs),
            "max": max(vs),
            "pass_rate_at_threshold": (
                sum(1 for v in vs if v >= thresholds[comp]) / len(vs)
                if comp in thresholds else None
            ),
        }
    return out
