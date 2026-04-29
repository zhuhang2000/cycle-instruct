from __future__ import annotations

from pathlib import Path

from code.I2QA import verify_cycle_consistency as verifier
from tool.multimodal_types import MultimodalInferConfig, VQAPair


def test_reconstruct_questions_uses_image_and_answer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    vqa = VQAPair(
        image_path="images/cat.png",
        image_id="cat",
        question="What animal is sitting on the chair?",
        answer="A cat is sitting on the chair.",
    )

    def fake_generate_multimodal(samples, build_messages, output_path, to_record, cfg):
        captured["samples"] = samples
        captured["output_path"] = output_path
        captured["cfg"] = cfg

        records = []
        for sample in samples:
            messages, image_paths = build_messages(sample)
            captured["messages"] = messages
            captured["image_paths"] = image_paths
            records.append(to_record(sample, "What animal is sitting on the chair?"))
        return records

    monkeypatch.setattr(verifier, "generate_multimodal", fake_generate_multimodal)

    cfg = MultimodalInferConfig(
        backend="hf",
        mllm_model_path="generator-model",
        verifier_model_path="verifier-model",
        max_new_tokens=64,
    )

    out = verifier.reconstruct_questions([vqa], cfg, tmp_path)

    assert out == ["What animal is sitting on the chair?"]
    assert captured["output_path"] == tmp_path / "question_reconstruction.json"

    samples = captured["samples"]
    assert len(samples) == 1
    assert samples[0].image_path == "images/cat.png"
    assert samples[0].source_text == "A cat is sitting on the chair."

    messages = captured["messages"]
    assert messages[1]["content"].startswith("<image>")
    assert "Target answer: A cat is sitting on the chair." in messages[1]["content"]
    assert captured["image_paths"] == ["images/cat.png"]

    mm_cfg = captured["cfg"]
    assert mm_cfg.backend == "hf"
    assert mm_cfg.model_path == "verifier-model"
    assert mm_cfg.mllm_model_path == "verifier-model"
