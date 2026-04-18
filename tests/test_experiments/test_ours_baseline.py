from __future__ import annotations

import json
from pathlib import Path

from experiments.baselines.ours import OursPreparer
from experiments.types import BaselineSpec


def test_ours_round_path_stays_under_configured_iterative_run_dir(tmp_path: Path) -> None:
    latest_dir = tmp_path / "runs" / "iterative" / "latest"
    round_dir = latest_dir / "round_1"
    round_dir.mkdir(parents=True)
    source = round_dir / "mixed_round_1.json"
    source.write_text(
        json.dumps([{"question": "q", "answer": "a", "image_path": "img.jpg"}]),
        encoding="utf-8",
    )

    spec = BaselineSpec(
        name="ours_round1",
        kind="ours",
        raw_pool_path=str(latest_dir),
        params={"round": 1},
    )

    out = OursPreparer(spec).prepare(tmp_path / "prepared")

    assert out.exists()
