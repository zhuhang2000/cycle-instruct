"""external_dataset baseline: use a pre-built third-party dataset verbatim.

Used for ShareGPT4V / LLaVA-Instruct / human-annotated subsets. The dataset
file is expected to already be in ShareGPT format (or a plain list we can
pass through ``to_sharegpt``).
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.baselines import register_baseline
from experiments.baselines.base import BaseDataPreparer, write_sharegpt


@register_baseline("external_dataset")
class ExternalDatasetPreparer(BaseDataPreparer):
    def prepare(self, output_dir: Path) -> Path:
        src = Path(self.spec.dataset_path)
        if not src.exists():
            raise FileNotFoundError(f"external dataset not found: {src}")
        if src.suffix == ".jsonl":
            from experiments.baselines.base import load_jsonl_pool
            data = load_jsonl_pool(src)
        else:
            data = json.loads(src.read_text("utf-8"))
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
        if self.spec.target_size and len(data) > self.spec.target_size:
            data = data[: self.spec.target_size]
        return write_sharegpt(data, Path(output_dir) / f"train_{self.spec.name}.json")
