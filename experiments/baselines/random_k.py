"""random_k baseline: uniform random subset of the raw pool.

``params.ratio`` (0<r<=1) OR ``params.k`` absolute count — if both are set,
``k`` wins. Seed is taken from ``BaselineSpec.seed`` for reproducibility.
"""
from __future__ import annotations

from pathlib import Path

from experiments.baselines import register_baseline
from experiments.baselines.base import (
    BaseDataPreparer,
    load_jsonl_pool,
    seeded_sample,
    write_sharegpt,
)


@register_baseline("random_k")
class RandomKPreparer(BaseDataPreparer):
    def prepare(self, output_dir: Path) -> Path:
        pool = load_jsonl_pool(Path(self.spec.raw_pool_path))
        k = self.spec.params.get("k")
        if k is None:
            ratio = float(self.spec.params.get("ratio", 0.5))
            k = max(1, int(ratio * len(pool)))
        k = min(int(k), self.spec.target_size or int(k))
        chosen = seeded_sample(pool, k, self.spec.seed)
        return write_sharegpt(chosen, Path(output_dir) / "train_random_k.json")
