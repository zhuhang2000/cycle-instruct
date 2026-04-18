"""clip_only baseline: retain samples whose CLIP(image, answer) ≥ threshold.

Each record in the raw pool is expected to carry a pre-computed
``cycle_scores.clip`` field. If it is absent, the record is dropped (we do
not recompute CLIP here — that's expensive and orthogonal to the baseline).
"""
from __future__ import annotations

from pathlib import Path

from experiments.baselines import register_baseline
from experiments.baselines.base import (
    BaseDataPreparer,
    load_jsonl_pool,
    write_sharegpt,
)


def _clip_score(sample: dict) -> float | None:
    cs = sample.get("cycle_scores") or {}
    v = cs.get("clip")
    return float(v) if v is not None else None


@register_baseline("clip_only")
class CLIPOnlyPreparer(BaseDataPreparer):
    def prepare(self, output_dir: Path) -> Path:
        threshold = float(self.spec.params.get("clip_threshold", 0.2))
        pool = load_jsonl_pool(Path(self.spec.raw_pool_path))
        kept = [r for r in pool if (s := _clip_score(r)) is not None and s >= threshold]
        if self.spec.target_size and len(kept) > self.spec.target_size:
            kept.sort(key=lambda r: -(_clip_score(r) or 0.0))
            kept = kept[: self.spec.target_size]
        return write_sharegpt(kept, Path(output_dir) / "train_clip_only.json")
