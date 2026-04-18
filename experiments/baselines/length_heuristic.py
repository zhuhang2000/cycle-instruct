"""length_heuristic baseline: keep records whose answer length lies in a band.

Very short answers are often degenerate ("yes"); very long answers are often
hallucinated prose. Params:
  - ``min_chars`` (default 3)
  - ``max_chars`` (default 300)
"""
from __future__ import annotations

from pathlib import Path

from experiments.baselines import register_baseline
from experiments.baselines.base import (
    BaseDataPreparer,
    load_jsonl_pool,
    write_sharegpt,
)


def _answer_text(sample: dict) -> str:
    if "messages" in sample:
        for msg in sample["messages"]:
            if msg.get("role") == "assistant":
                return str(msg.get("content", ""))
    return str(sample.get("answer", ""))


@register_baseline("length_heuristic")
class LengthHeuristicPreparer(BaseDataPreparer):
    def prepare(self, output_dir: Path) -> Path:
        lo = int(self.spec.params.get("min_chars", 3))
        hi = int(self.spec.params.get("max_chars", 300))
        pool = load_jsonl_pool(Path(self.spec.raw_pool_path))
        kept = [r for r in pool if lo <= len(_answer_text(r).strip()) <= hi]
        if self.spec.target_size and len(kept) > self.spec.target_size:
            kept = kept[: self.spec.target_size]
        return write_sharegpt(kept, Path(output_dir) / "train_length_heuristic.json")
