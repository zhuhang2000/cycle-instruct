"""Human pairwise-preference evaluation: CSV template + inter-rater agreement.

Typical workflow:
1. ``generate_template(samples_a, samples_b, output_csv)`` — creates a CSV
   where each row is a sample pair with blank ``rater_1`` / ``rater_2`` columns.
2. Raters fill in "A"/"B"/"tie".
3. ``analyse(csv_path)`` returns win-rates and Cohen's κ between two raters.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalise_label(value: str) -> str | None:
    v = value.strip().lower()
    return v if v in ("a", "b", "tie") else None


def generate_template(
    samples_a: list[dict[str, Any]],
    samples_b: list[dict[str, Any]],
    output_csv: Path,
    rater_count: int = 2,
) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(samples_a), len(samples_b))
    fieldnames = ["pair_id", "question_a", "answer_a", "question_b", "answer_b"]
    fieldnames += [f"rater_{i+1}" for i in range(rater_count)]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n):
            a = samples_a[i]
            b = samples_b[i]
            w.writerow({
                "pair_id": i,
                "question_a": a.get("question", ""),
                "answer_a": a.get("answer", ""),
                "question_b": b.get("question", ""),
                "answer_b": b.get("answer", ""),
                **{f"rater_{j+1}": "" for j in range(rater_count)},
            })
    return output_csv


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's κ for two equal-length label lists (any string labels)."""
    assert len(a) == len(b) and a
    labels = sorted(set(a) | set(b))
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if abs(1 - pe) < 1e-12:
        return 1.0 if agree == 1.0 else 0.0
    return (agree - pe) / (1 - pe)


def analyse(csv_path: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(Path(csv_path).open("r", encoding="utf-8")))
    rater_cols = [k for k in rows[0].keys() if k.startswith("rater_")] if rows else []
    judgments: dict[str, list[str]] = {c: [] for c in rater_cols}
    for r in rows:
        for c in rater_cols:
            v = _normalise_label(r.get(c) or "")
            if v is not None:
                judgments[c].append(v)

    summary: dict[str, Any] = {
        "num_pairs": len(rows),
        "rater_win_rates": {},
        "kappas": {},
    }
    for c, labels in judgments.items():
        n = len(labels) or 1
        summary["rater_win_rates"][c] = {
            "A": labels.count("a") / n,
            "B": labels.count("b") / n,
            "tie": labels.count("tie") / n,
        }
    if len(rater_cols) >= 2:
        c1, c2 = rater_cols[0], rater_cols[1]
        common = []
        for row in rows:
            x = _normalise_label(row.get(c1) or "")
            y = _normalise_label(row.get(c2) or "")
            if x is not None and y is not None:
                common.append((x, y))
        if common:
            a_list, b_list = zip(*common)
            summary["kappas"][f"{c1}_vs_{c2}"] = cohens_kappa(list(a_list), list(b_list))
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template")
    t.add_argument("--samples-a", type=Path, required=True)
    t.add_argument("--samples-b", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)

    a = sub.add_parser("analyse")
    a.add_argument("csv", type=Path)

    args = ap.parse_args(argv)
    if args.cmd == "template":
        import json
        sa = json.loads(args.samples_a.read_text("utf-8"))
        sb = json.loads(args.samples_b.read_text("utf-8"))
        generate_template(sa, sb, args.output)
    else:
        import json
        print(json.dumps(analyse(args.csv), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
