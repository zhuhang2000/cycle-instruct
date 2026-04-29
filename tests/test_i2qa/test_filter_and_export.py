from __future__ import annotations

import json
from pathlib import Path

from code.I2QA.filter_and_export import filter_and_export
from tool.multimodal_types import MultimodalInferConfig, VQAPair


def test_filter_and_export_preserves_cycle_scores(tmp_path: Path) -> None:
    output_path = tmp_path / "filtered.json"
    vqa = VQAPair(
        image_path="images/chart.png",
        image_id="chart",
        question="What does the chart show?",
        answer="The chart shows sales increasing.",
        generation_model="mock-vlm",
        cycle_scores={
            "composite": 0.91,
            "ar": 0.88,
            "clip": 0.42,
            "qr": 0.81,
            "ppl": 0.5,
        },
    )

    records = filter_and_export(
        [vqa],
        MultimodalInferConfig(
            cycle_threshold=0.7,
            ar_threshold=0.6,
            clip_threshold=0.2,
            qr_threshold=0.55,
        ),
        output_path,
    )

    assert len(records) == 1
    record = records[0]
    assert record["messages"][0]["content"] == "<image>What does the chart show?"
    assert record["images"] == ["images/chart.png"]
    assert record["cycle_scores"]["composite"] == 0.91
    assert record["cycle_score"] == 0.91
    assert record["image_id"] == "chart"
    assert record["generation_model"] == "mock-vlm"

    saved = json.loads(output_path.read_text("utf-8"))
    assert saved == records
