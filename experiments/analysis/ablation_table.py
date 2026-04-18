"""Ablation table: same layout as main table but adds Δ vs. the "full" method.

Expected convention: an ablation run-root contains methods named
``full``, ``drop_ar``, ``drop_clip``, ``drop_qr``, ``drop_ppl``. The Δ is
computed relative to ``full`` for every benchmark.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.analysis.aggregate_tables import (
    build_grid,
    collect_runs,
    _fmt,
)


def _delta_cell(full_score: float | None, score: float | None) -> str:
    if full_score is None or score is None:
        return "--"
    return f"{score:.3f} ({score - full_score:+.3f})"


def aggregate_ablation(experiment_dir: Path, output_dir: Path,
                      full_method: str = "full") -> dict[str, Path]:
    runs = collect_runs(experiment_dir)
    methods, benchmarks, cells = build_grid(runs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if full_method not in methods:
        # No reference: fall back to main-table formatting.
        md = ("| Method | " + " | ".join(benchmarks) + " |\n"
              "|" + "---|" * (len(benchmarks) + 1) + "\n")
        for m in methods:
            md += "| " + " | ".join(
                [m] + [_fmt((cells.get((m, b)) or {}).get("score")) for b in benchmarks]
            ) + " |\n"
    else:
        full_scores = {b: (cells.get((full_method, b)) or {}).get("score")
                       for b in benchmarks}
        md_lines = ["| Method | " + " | ".join(benchmarks) + " |",
                    "|" + "---|" * (len(benchmarks) + 1)]
        for m in methods:
            row = [m]
            for b in benchmarks:
                s = (cells.get((m, b)) or {}).get("score")
                row.append(
                    _fmt(s) if m == full_method
                    else _delta_cell(full_scores.get(b), s)
                )
            md_lines.append("| " + " | ".join(row) + " |")
        md = "\n".join(md_lines) + "\n"

    md_path = output_dir / "ablation_table.md"
    md_path.write_text(md, encoding="utf-8")
    return {"md": md_path}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("--full-method", default="full")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output_dir or (args.experiment_dir / "paper_tables")
    aggregate_ablation(args.experiment_dir, out, full_method=args.full_method)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
