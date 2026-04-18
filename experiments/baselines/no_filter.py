"""no_filter baseline: pass the raw generation pool through as-is.

Useful as a lower bound — demonstrates that unfiltered MLLM output is noisy.
"""
from __future__ import annotations

from pathlib import Path

from experiments.baselines import register_baseline
from experiments.baselines.base import (
    BaseDataPreparer,
    load_jsonl_pool,
    write_sharegpt,
)


@register_baseline("no_filter")
class NoFilterPreparer(BaseDataPreparer):
    def prepare(self, output_dir: Path) -> Path:
        pool = load_jsonl_pool(Path(self.spec.raw_pool_path))
        if self.spec.target_size and len(pool) > self.spec.target_size:
            pool = pool[: self.spec.target_size]
        return write_sharegpt(pool, Path(output_dir) / "train_no_filter.json")
