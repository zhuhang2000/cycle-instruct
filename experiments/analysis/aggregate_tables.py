"""Scan runs root → produce main table (method × benchmark) in md/csv/tex.

Each ``runs/experiments/<spec>/<method>/run.json`` is loaded and turned into a
2-D cell grid. Missing cells render as ``--``. The md variant is
copy-paste-ready for a paper draft; the tex variant uses booktabs rules.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from experiments.types import load_json, save_json

logger = logging.getLogger(__name__)


def collect_runs(experiment_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for run_json in sorted(Path(experiment_dir).glob("*/run.json")):
        try:
            runs.append(load_json(run_json))
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load %s: %s", run_json, exc)
    return runs


def build_grid(runs: list[dict]) -> tuple[list[str], list[str], dict[tuple[str, str], dict[str, Any]]]:
    """Return (methods, benchmarks, cells)."""
    methods: list[str] = []
    benchmarks: list[str] = []
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        m = run.get("method", "unknown")
        if m not in methods:
            methods.append(m)
        for r in run.get("results", []):
            b = r.get("benchmark", "")
            if b and b not in benchmarks:
                benchmarks.append(b)
            cells[(m, b)] = {
                "score": r.get("score"),
                "metric": r.get("metric"),
                "num_samples": r.get("num_samples"),
            }
    return methods, benchmarks, cells


def _fmt(v: Any) -> str:
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def to_markdown(methods: list[str], benchmarks: list[str],
                cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    head = "| Method | " + " | ".join(benchmarks) + " |"
    sep = "|" + "---|" * (len(benchmarks) + 1)
    lines = [head, sep]
    for m in methods:
        row = [m] + [_fmt((cells.get((m, b)) or {}).get("score")) for b in benchmarks]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def to_csv(methods: list[str], benchmarks: list[str],
           cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    lines = ["method," + ",".join(benchmarks)]
    for m in methods:
        row = [m] + [_fmt((cells.get((m, b)) or {}).get("score")) for b in benchmarks]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def to_latex(methods: list[str], benchmarks: list[str],
             cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    col_spec = "l" + "c" * len(benchmarks)
    out = [r"\begin{tabular}{" + col_spec + "}", r"\toprule",
           "Method & " + " & ".join(benchmarks) + r" \\", r"\midrule"]
    for m in methods:
        row = [m.replace("_", r"\_")] + [
            _fmt((cells.get((m, b)) or {}).get("score")) for b in benchmarks
        ]
        out.append(" & ".join(row) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out) + "\n"


def aggregate(experiment_dir: Path, output_dir: Path) -> dict[str, Path]:
    runs = collect_runs(experiment_dir)
    methods, benchmarks, cells = build_grid(runs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "main_table.md"
    csv_path = output_dir / "main_table.csv"
    tex_path = output_dir / "main_table.tex"
    md_path.write_text(to_markdown(methods, benchmarks, cells), encoding="utf-8")
    csv_path.write_text(to_csv(methods, benchmarks, cells), encoding="utf-8")
    tex_path.write_text(to_latex(methods, benchmarks, cells), encoding="utf-8")

    summary = [
        {"method": m, "benchmark": b,
         "score": (cells.get((m, b)) or {}).get("score"),
         "metric": (cells.get((m, b)) or {}).get("metric"),
         "num_samples": (cells.get((m, b)) or {}).get("num_samples")}
        for m in methods for b in benchmarks
    ]
    save_json(output_dir / "summary.json", summary)

    logger.info("aggregated %d methods × %d benchmarks -> %s",
                len(methods), len(benchmarks), output_dir)
    return {"md": md_path, "csv": csv_path, "tex": tex_path,
            "summary": output_dir / "summary.json"}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output_dir or (args.experiment_dir / "paper_tables")
    aggregate(args.experiment_dir, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
