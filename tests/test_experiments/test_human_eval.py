from __future__ import annotations

import csv
from pathlib import Path

from experiments.analysis.human_eval import analyse


def test_analyse_preserves_row_alignment_when_some_ratings_are_blank(tmp_path: Path) -> None:
    csv_path = tmp_path / "ratings.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair_id", "question_a", "answer_a", "question_b", "answer_b", "rater_1", "rater_2"],
        )
        writer.writeheader()
        writer.writerow({"pair_id": 0, "rater_1": "a", "rater_2": "a"})
        writer.writerow({"pair_id": 1, "rater_1": "", "rater_2": "a"})
        writer.writerow({"pair_id": 2, "rater_1": "b", "rater_2": "b"})

    summary = analyse(csv_path)

    assert summary["num_pairs"] == 3
    assert summary["kappas"]["rater_1_vs_rater_2"] == 1.0
