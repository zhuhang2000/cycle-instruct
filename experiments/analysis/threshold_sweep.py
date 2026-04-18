"""Threshold sweep: plot primary-metric vs. filter threshold θ.

Each subdir of ``sweep_dir`` is a run at a specific θ, named ``theta_0p60``,
``theta_0p70`` etc. We pull the primary score per benchmark across thetas
and emit one PNG per benchmark.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from experiments.types import load_json, save_json

logger = logging.getLogger(__name__)


_THETA_RE = re.compile(r"theta[_-]?(?P<val>\d+p\d+|\d+\.\d+)")


def _parse_theta(name: str) -> float | None:
    m = _THETA_RE.search(name)
    if not m:
        return None
    v = m.group("val").replace("p", ".")
    try:
        return float(v)
    except ValueError:
        return None


def collect_sweep(sweep_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """Return ``{benchmark_name: [(theta, score), ...]}`` sorted by theta."""
    series: dict[str, list[tuple[float, float]]] = {}
    for child in sorted(Path(sweep_dir).iterdir()):
        if not child.is_dir():
            continue
        theta = _parse_theta(child.name)
        if theta is None:
            continue
        run_json = child / "run.json"
        if not run_json.exists():
            continue
        data = load_json(run_json)
        for r in data.get("results", []):
            b = r["benchmark"]
            series.setdefault(b, []).append((theta, float(r["score"])))
    for b in series:
        series[b].sort(key=lambda x: x[0])
    return series


def plot_sweep(series: dict[str, list[tuple[float, float]]], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib missing; skipping plot")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for b, pts in series.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=b)
    ax.set_xlabel(r"Filter threshold $\theta$")
    ax.set_ylabel("Benchmark score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_sweep.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output_dir or (args.sweep_dir / "paper_tables")
    series = collect_sweep(args.sweep_dir)
    save_json(out / "threshold_sweep.json", series)
    plot_sweep(series, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
