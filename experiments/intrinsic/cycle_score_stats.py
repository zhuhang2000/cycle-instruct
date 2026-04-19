"""Cycle-score descriptive statistics.

Produces mean/std/percentile summaries, a Pearson correlation matrix
across components, and pass-rate-vs-threshold curves used by the
threshold-sensitivity figure in the paper.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from experiments.intrinsic._io import load_vqa_jsonl, save_json
from experiments.intrinsic.base import IntrinsicMetric, register_metric


_COMPONENTS = ("ar", "clip", "qr", "ppl", "composite")
_DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
_PERCENTILES = [25, 50, 75, 90, 95, 99]


def _collect(samples: Iterable[dict], comp: str) -> list[float]:
    out: list[float] = []
    for s in samples:
        cs = s.get("cycle_scores") or {}
        v = cs.get(comp)
        if v is not None:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
    return out


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile over an unsorted list."""
    if not values:
        return math.nan
    vs = sorted(values)
    k = max(0, min(len(vs) - 1, int(round((q / 100) * (len(vs) - 1)))))
    return vs[k]


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denom_x = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    denom_y = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return num / (denom_x * denom_y)


def _histogram(values: list[float], bins: int = 50) -> dict[str, list[float]]:
    if not values:
        return {"edges": [], "counts": []}
    lo, hi = min(values), max(values)
    if lo == hi:
        return {"edges": [lo, hi], "counts": [len(values)]}
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    return {"edges": edges, "counts": counts}


def _component_values(samples: list[dict]) -> dict[str, list[float]]:
    return {c: _collect(samples, c) for c in _COMPONENTS}


@register_metric("cycle_stats")
class CycleScoreStats(IntrinsicMetric):
    """Module-4 metric: statistics over ``cycle_scores`` fields."""

    requires_cycle_scores = True

    def compute(
        self,
        samples: list[dict],
        *,
        thresholds: list[float] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        thresholds = thresholds or _DEFAULT_THRESHOLDS
        values = _component_values(samples)

        components: dict[str, dict[str, Any]] = {}
        for comp, vs in values.items():
            if not vs:
                components[comp] = {"count": 0}
                continue
            components[comp] = {
                "count": len(vs),
                "mean": mean(vs),
                "std": pstdev(vs) if len(vs) > 1 else 0.0,
                "min": min(vs),
                "max": max(vs),
                **{f"p{q}": percentile(vs, q) for q in _PERCENTILES},
                "histogram": _histogram(vs),
            }

        pass_rate: dict[str, dict[float, float]] = {}
        comp_vs = values.get("composite") or []
        for t in thresholds:
            key = f"{t:.2f}"
            pr = (sum(1 for v in comp_vs if v >= t) / len(comp_vs)) if comp_vs else 0.0
            pass_rate.setdefault("composite", {})[key] = pr

        # Pairwise correlations, restricted to samples where BOTH components exist.
        correlations: dict[str, dict[str, float]] = {}
        names = list(_COMPONENTS)
        # Build per-sample component vectors for alignment.
        aligned = {c: [] for c in names}
        for s in samples:
            cs = s.get("cycle_scores") or {}
            row = {c: cs.get(c) for c in names}
            if all(isinstance(row[c], (int, float)) for c in names):
                for c in names:
                    aligned[c].append(float(row[c]))
        for a in names:
            correlations[a] = {}
            for b in names:
                correlations[a][b] = _pearson(aligned[a], aligned[b])

        return {
            "num_samples": len(samples),
            "components": components,
            "pass_rate_at_threshold": pass_rate,
            "component_correlations": correlations,
            "aligned_count": len(aligned[names[0]]),
        }

    def plots(self, result: dict, out_dir: Path) -> list[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return []
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        comp = result.get("components", {}).get("composite", {})
        hist = comp.get("histogram") if comp else None
        if hist and hist.get("counts"):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(range(len(hist["counts"])), hist["counts"])
            ax.set_title("Composite cycle-score histogram")
            path = out_dir / "composite_score_hist.png"
            fig.tight_layout()
            fig.savefig(path, dpi=110)
            plt.close(fig)
            paths.append(path)

        pr = result.get("pass_rate_at_threshold", {}).get("composite") or {}
        if pr:
            xs = sorted(float(k) for k in pr.keys())
            ys = [pr[f"{x:.2f}"] for x in xs]
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(xs, ys, marker="o")
            ax.set_xlabel("threshold")
            ax.set_ylabel("pass rate")
            ax.set_title("Pass rate vs composite threshold")
            path = out_dir / "pass_rate_vs_threshold.png"
            fig.tight_layout()
            fig.savefig(path, dpi=110)
            plt.close(fig)
            paths.append(path)

        return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cycle-score stats.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    samples = load_vqa_jsonl(args.input)
    result = CycleScoreStats().compute(samples)
    if args.out:
        save_json(args.out, result)
    else:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
