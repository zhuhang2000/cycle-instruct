"""QA type distribution: counts, Shannon entropy, JS-divergence vs seed.

Delegates classification to :mod:`code.iterative.qa_templates` so stats
here stay consistent with the generator-side rebalancer.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments.intrinsic._io import (
    extract_question,
    load_vqa_jsonl,
    save_json,
)
from experiments.intrinsic.base import IntrinsicMetric, register_metric


def _qa_templates_api():
    """Lazy import so a missing ``code.iterative`` doesn't break module load."""
    from code.iterative.qa_templates import (  # noqa: WPS433
        QA_TYPES,
        classify_qa_type,
        compute_diversity_score,
        compute_type_distribution,
    )
    return QA_TYPES, classify_qa_type, compute_diversity_score, compute_type_distribution


def _normalize(sample: dict) -> dict:
    """Ensure ``question`` field is populated for qa_templates code path."""
    if sample.get("question"):
        return sample
    return {**sample, "question": extract_question(sample)}


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence between two discrete distributions.

    Symmetric, non-negative, bounded by ``log 2``. Keys missing from
    either input are treated as zero probability.
    """
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def _kl(a: dict[str, float], b: dict[str, float]) -> float:
        total = 0.0
        for k, pa in a.items():
            pb = b.get(k, 0.0)
            if pa > 0 and pb > 0:
                total += pa * math.log(pa / pb)
        return total

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def per_type_breakdown(
    samples: Iterable[dict],
    *,
    composite_threshold: float = 0.7,
) -> dict[str, dict[str, float]]:
    """Per-type mean cycle score + pass-rate at a composite threshold."""
    _, classify_qa_type, _, _ = _qa_templates_api()
    buckets: dict[str, list[float]] = {}
    passes: dict[str, int] = {}
    for s in samples:
        q = extract_question(s)
        if not q:
            continue
        t = classify_qa_type(q)
        cs = s.get("cycle_scores") or {}
        comp = cs.get("composite")
        if comp is None:
            continue
        try:
            comp = float(comp)
        except (TypeError, ValueError):
            continue
        buckets.setdefault(t, []).append(comp)
        passes[t] = passes.get(t, 0) + (1 if comp >= composite_threshold else 0)
    out: dict[str, dict[str, float]] = {}
    for t, scores in buckets.items():
        n = len(scores)
        out[t] = {
            "count": n,
            "mean_composite": sum(scores) / n,
            "pass_rate": passes.get(t, 0) / n,
        }
    return out


@register_metric("qa_types")
class QaTypeStats(IntrinsicMetric):
    """Module-1 metric: QA type distribution + entropy + JS divergence."""

    def compute(
        self,
        samples: list[dict],
        *,
        seed_ref: list[dict] | None = None,
        min_fraction: float = 0.05,
        composite_threshold: float = 0.7,
        **_: Any,
    ) -> dict[str, Any]:
        QA_TYPES, _, compute_diversity_score, compute_type_distribution = _qa_templates_api()

        norm = [_normalize(s) for s in samples]
        dist = compute_type_distribution(norm)
        diversity = compute_diversity_score(dist)

        result: dict[str, Any] = {
            "num_samples": len(samples),
            "type_distribution": {t: dist.get(t, 0.0) for t in QA_TYPES},
            "diversity_score": diversity,
            "under_represented_types": [
                t for t in QA_TYPES if dist.get(t, 0.0) < min_fraction
            ],
            "per_type": per_type_breakdown(norm, composite_threshold=composite_threshold),
        }

        if seed_ref is not None:
            seed_norm = [_normalize(s) for s in seed_ref]
            seed_dist = compute_type_distribution(seed_norm)
            result["seed_type_distribution"] = {
                t: seed_dist.get(t, 0.0) for t in QA_TYPES
            }
            result["js_divergence_vs_seed"] = js_divergence(dist, seed_dist)
        return result

    def plots(self, result: dict, out_dir: Path) -> list[Path]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return []
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dist = result.get("type_distribution", {})
        if not dist:
            return []
        fig, ax = plt.subplots(figsize=(6, 3.5))
        labels = list(dist.keys())
        vals = [dist[k] for k in labels]
        ax.bar(labels, vals)
        ax.set_ylabel("Fraction")
        ax.set_title("QA type distribution")
        ax.tick_params(axis="x", rotation=30)
        path = out_dir / "qa_type_distribution.png"
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return [path]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="QA type distribution stats.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--seed-ref", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min-fraction", type=float, default=0.05)
    args = p.parse_args(argv)

    samples = load_vqa_jsonl(args.input)
    seed = load_vqa_jsonl(args.seed_ref) if args.seed_ref else None
    result = QaTypeStats().compute(samples, seed_ref=seed, min_fraction=args.min_fraction)
    if args.out:
        save_json(args.out, result)
    else:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
