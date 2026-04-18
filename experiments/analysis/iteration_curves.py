"""Plot round-level metrics from an iterative run.

Reads ``runs/iterative/<run>/round_*/metrics.json`` and produces a 2×2 panel
of (pass_rate / mean_cycle / data_diversity / eval_accuracy) vs. round_id.

matplotlib is optional; if missing, the raw JSON is still written.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from experiments.types import load_json, save_json

logger = logging.getLogger(__name__)


def collect_round_metrics(run_dir: Path) -> list[dict[str, Any]]:
    rounds = []
    for md in sorted(Path(run_dir).glob("round_*/metrics.json")):
        rounds.append(load_json(md))
    rounds.sort(key=lambda r: r.get("round_id", 0))
    return rounds


def plot_curves(rounds: list[dict[str, Any]], output_path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot")
        return None

    if not rounds:
        logger.warning("no rounds to plot")
        return None

    xs = [r.get("round_id", i) for i, r in enumerate(rounds)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, key, title in [
        (axes[0, 0], "pass_rate", "Pass Rate"),
        (axes[0, 1], "mean_cycle_score", "Mean Cycle Score"),
        (axes[1, 0], "data_diversity_score", "Diversity Score"),
        (axes[1, 1], "eval_accuracy", "Eval Accuracy"),
    ]:
        ys = [r.get(key) for r in rounds]
        ax.plot(xs, ys, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output_dir or (args.run_dir / "figures")
    rounds = collect_round_metrics(args.run_dir)
    save_json(out / "iteration_curves.json", rounds)
    plot_curves(rounds, out / "iteration_curves.png")
    logger.info("wrote %d round(s) -> %s", len(rounds), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
