"""Base interface for data-preparation baselines.

A baseline is a deterministic function
    ``BaselineSpec -> ShareGPT-formatted training JSON``
that the runner can call without touching a GPU. The output path is what
``baselines/runner.py`` then feeds to the training hook.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from experiments.types import BaselineSpec, save_json


class BaseDataPreparer:
    """Contract for all baseline data preparers."""

    baseline_name: str = ""

    def __init__(self, spec: BaselineSpec) -> None:
        self.spec = spec

    def prepare(self, output_dir: Path) -> Path:
        """Produce a ShareGPT-formatted JSON file. Return its path."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_jsonl_pool(path: Path) -> list[dict]:
    out: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def to_sharegpt(sample: dict) -> dict:
    """Normalise a VQA record to ShareGPT schema.

    Accepts either:
      - Already-ShareGPT: ``{"messages": [...], "images": [...]}``
      - Legacy VQAPair-style: ``{"image_path", "question", "answer"}``
    """
    if "messages" in sample and "images" in sample:
        return {"messages": sample["messages"], "images": sample["images"]}
    q = sample.get("question", "")
    a = sample.get("answer", "")
    img = sample.get("image_path") or sample.get("image")
    return {
        "messages": [
            {"role": "user", "content": f"<image>{q}"},
            {"role": "assistant", "content": a},
        ],
        "images": [img] if img else [],
    }


def write_sharegpt(records: Iterable[dict], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, [to_sharegpt(r) for r in records])
    return output_path


def seeded_sample(pool: list[Any], k: int, seed: int) -> list[Any]:
    if k >= len(pool):
        return list(pool)
    rng = random.Random(seed)
    return rng.sample(pool, k)
