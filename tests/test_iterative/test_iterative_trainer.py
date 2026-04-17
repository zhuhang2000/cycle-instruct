"""Tests for the iterative controller's Stage 1-3 subprocess wiring."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from code.iterative import iterative_trainer


def test_run_generation_and_filter_uses_current_i2qa_cli(
    monkeypatch, tmp_path: Path,
) -> None:
    raw_image_dir = tmp_path / "images"
    raw_image_dir.mkdir()
    for name in ["a.png", "b.jpg"]:
        (raw_image_dir / name).write_bytes(b"fake-image")

    round_dir = tmp_path / "round_0"
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)

        output_path = Path(cmd[cmd.index("--output") + 1])
        script_name = Path(cmd[1]).name
        if script_name == "generate_vqa_pairs.py":
            output_path.write_text(
                json.dumps(
                    [
                        {
                            "image_path": "img.png",
                            "image_id": "img",
                            "question": "What is shown?",
                            "answer": "A chart.",
                            "generation_model": "mock",
                            "cycle_scores": {},
                        }
                    ]
                ),
                encoding="utf-8",
            )
        elif script_name == "verify_cycle_consistency.py":
            output_path.write_text(
                json.dumps(
                    [
                        {
                            "image_path": "img.png",
                            "image_id": "img",
                            "question": "What is shown?",
                            "answer": "A chart.",
                            "generation_model": "mock",
                            "cycle_scores": {
                                "composite": 0.9,
                                "ar": 0.8,
                                "clip": 0.4,
                                "qr": 0.7,
                                "ppl": 0.5,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
        elif script_name == "filter_and_export.py":
            output_path.write_text(
                json.dumps(
                    [
                        {
                            "messages": [
                                {"role": "user", "content": "<image>What is shown?"},
                                {"role": "assistant", "content": "A chart."},
                            ],
                            "images": ["img.png"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(iterative_trainer.subprocess, "run", fake_run)

    filtered = iterative_trainer.run_generation_and_filter(
        generator_model_path="/models/mock-mllm",
        raw_image_dir=str(raw_image_dir),
        round_dir=round_dir,
        samples_to_generate=5,
    )

    assert filtered == [
        {
            "messages": [
                {"role": "user", "content": "<image>What is shown?"},
                {"role": "assistant", "content": "A chart."},
            ],
            "images": ["img.png"],
        }
    ]
    assert len(calls) == 3

    generate_cmd = calls[0]
    assert generate_cmd[0] == sys.executable
    assert Path(generate_cmd[1]).name == "generate_vqa_pairs.py"
    assert "--input" in generate_cmd
    assert "--model-path" in generate_cmd
    assert "--num-qa" in generate_cmd
    assert "--model_path" not in generate_cmd
    assert "--image_dir" not in generate_cmd
    assert "--num_samples" not in generate_cmd
    assert generate_cmd[generate_cmd.index("--num-qa") + 1] == "1"

    stage1_input = Path(generate_cmd[generate_cmd.index("--input") + 1])
    records = json.loads(stage1_input.read_text(encoding="utf-8"))
    assert len(records) == 5
    assert {Path(record["image_path"]).name for record in records} == {"a.png", "b.jpg"}

    verify_cmd = calls[1]
    assert Path(verify_cmd[1]).name == "verify_cycle_consistency.py"
    for flag in ("--model-path", "--verifier-model-path", "--text-model-path"):
        assert verify_cmd[verify_cmd.index(flag) + 1] == "/models/mock-mllm"

    filter_cmd = calls[2]
    assert Path(filter_cmd[1]).name == "filter_and_export.py"
