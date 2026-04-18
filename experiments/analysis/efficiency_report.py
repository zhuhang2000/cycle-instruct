"""GPU-hour and wall-time rollup across MethodRun JSONs."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.analysis.aggregate_tables import collect_runs
from experiments.types import save_json


def summarise(experiment_dir: Path) -> list[dict]:
    rows = []
    for run in collect_runs(experiment_dir):
        prep = run.get("prepare_time_sec", 0.0)
        train = run.get("train_time_sec", 0.0)
        ev = run.get("eval_time_sec", 0.0)
        rows.append({
            "method": run.get("method"),
            "prepare_sec": prep,
            "train_sec": train,
            "eval_sec": ev,
            "total_sec": prep + train + ev,
            "gpu_hours": run.get("gpu_hours", 0.0),
        })
    rows.sort(key=lambda r: r["total_sec"])
    return rows


def to_markdown(rows: list[dict]) -> str:
    head = "| Method | Prep (s) | Train (s) | Eval (s) | Total (s) | GPU-h |"
    sep = "|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['prepare_sec']:.1f} | {r['train_sec']:.1f} | "
            f"{r['eval_sec']:.1f} | {r['total_sec']:.1f} | {r['gpu_hours']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    rows = summarise(args.experiment_dir)
    out = args.output_dir or (args.experiment_dir / "paper_tables")
    out.mkdir(parents=True, exist_ok=True)
    (out / "efficiency.md").write_text(to_markdown(rows), encoding="utf-8")
    save_json(out / "efficiency.json", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
