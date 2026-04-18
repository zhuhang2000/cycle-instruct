from __future__ import annotations

from pathlib import Path

from experiments.baselines.runner import default_train_fn


def test_default_train_fn_uses_supported_run_multimodal_cycle_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, list[str]] = {}

    class _Proc:
        returncode = 0

    def _fake_run(cmd, stdout=None, stderr=None, check=None):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr("subprocess.run", _fake_run)

    train_fn = default_train_fn("/models/qwen-vl", preset="unused")
    dataset = tmp_path / "train.json"
    dataset.write_text("[]", encoding="utf-8")

    train_fn(dataset, tmp_path / "model")

    assert captured["cmd"] == [
        "bash",
        str(Path("bash/run_multimodal_cycle.sh").resolve()),
        "--input",
        str(dataset),
        "--output-dir",
        str(tmp_path / "model"),
        "--mllm-model",
        "/models/qwen-vl",
        "--data-type",
        "json",
    ]
