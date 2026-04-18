from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

from experiments.eval.runner import default_infer_fn


def test_default_infer_fn_uses_real_artifact_path_instead_of_dev_null(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Path] = {}

    def _fake_generate_multimodal(samples, build_messages, output_path, to_record, cfg):
        captured["output_path"] = output_path
        assert output_path != Path("/dev/null")
        assert tmp_path in output_path.parents
        return [{"raw": "model-output"}]

    @dataclass
    class _ImageTextSample:
        image_path: str
        image_id: str
        source_text: str | None
        source_type: str
        metadata: dict

    @dataclass
    class _MultimodalInferConfig:
        mllm_model_path: str = ""

    infer_mod = types.ModuleType("tool.multimodal_infer")
    infer_mod.generate_multimodal = _fake_generate_multimodal
    types_mod = types.ModuleType("tool.multimodal_types")
    types_mod.ImageTextSample = _ImageTextSample
    types_mod.MultimodalInferConfig = _MultimodalInferConfig

    monkeypatch.setitem(sys.modules, "tool.multimodal_infer", infer_mod)
    monkeypatch.setitem(sys.modules, "tool.multimodal_types", types_mod)

    infer_fn = default_infer_fn(output_root=tmp_path)
    out = infer_fn([{"role": "user", "content": "<image>q"}], ["img.jpg"], "model-path")

    assert out == "model-output"
    assert captured["output_path"].name.startswith("infer_")
