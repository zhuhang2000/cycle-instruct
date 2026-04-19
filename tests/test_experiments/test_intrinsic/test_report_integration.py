"""End-to-end report smoke: mock heavy detector, verify artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.intrinsic._io import load_vqa_jsonl
from experiments.intrinsic.base import METRIC_REGISTRY
from experiments.intrinsic.report import main as report_main, run_report


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "tiny_vqa.jsonl"


def test_registry_has_expected_modules():
    for name in ("qa_types", "diversity", "cycle_stats", "linguistic",
                 "hallucination", "alignment"):
        assert name in METRIC_REGISTRY, f"{name} missing from metric registry"


def test_run_report_writes_artifacts(tmp_path: Path) -> None:
    samples = load_vqa_jsonl(FIXTURE)
    modules = {
        "qa_types": {"enabled": True},
        "diversity": {"enabled": True, "n_sample_self_bleu": 10},
        "cycle_stats": {"enabled": True},
        "linguistic": {"enabled": True},
        # hallucination: enabled with injected no-op detector (no GPU)
        "hallucination": {
            "enabled": True,
            "detector": lambda img, names: {n: True for n in names},
            "use_spacy": False,
        },
    }
    report = run_report(samples, modules=modules, out_dir=tmp_path)

    assert (tmp_path / "intrinsic_report.json").exists()
    assert (tmp_path / "intrinsic_report.md").exists()

    raw = json.loads((tmp_path / "intrinsic_report.json").read_text("utf-8"))
    assert raw["num_samples"] == len(samples)
    assert raw["qa_types"]["num_samples"] == len(samples)
    assert raw["cycle_stats"]["num_samples"] == len(samples)
    # CHAIRi must be 0 since the detector always returns True
    assert raw["hallucination"]["chairi"] == 0.0

    md = (tmp_path / "intrinsic_report.md").read_text("utf-8")
    assert "QA type distribution" in md
    assert "Hallucination" in md


def test_cli_smoke_mode(tmp_path: Path) -> None:
    out = tmp_path / "report"
    exit_code = report_main([
        "--input", str(FIXTURE),
        "--out", str(out),
        "--smoke",
        "--modules", "qa_types,diversity,cycle_stats,linguistic",
    ])
    assert exit_code == 0
    assert (out / "intrinsic_report.json").exists()


def test_disabled_modules_are_skipped(tmp_path: Path) -> None:
    samples = load_vqa_jsonl(FIXTURE)
    modules = {
        "qa_types": {"enabled": False},
        "diversity": {"enabled": True, "n_sample_self_bleu": 5},
    }
    report = run_report(samples, modules=modules, out_dir=tmp_path)
    assert "qa_types" not in report
    assert "diversity" in report
