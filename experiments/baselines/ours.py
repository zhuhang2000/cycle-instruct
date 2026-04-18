"""Our method: use the iterative-pipeline output for a specific round.

Two convenient modes driven by ``spec.params``:

* ``params.round=N`` — load the filtered-and-mixed training JSON produced
  by round *N* of the iterative trainer. If the path is provided via
  ``params.round_data`` we use it directly; otherwise we look for
  ``<raw_pool_path>/round_{N}/mixed_round_{N}.json``.
* ``params.filtered_pool`` — treat the file as the raw filtered VQA set
  (single-shot cycle-consistency, no mixing).

Either way, the prepared file is re-written into ``output_dir`` so the
baseline runner has a self-contained artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.baselines import register_baseline
from experiments.baselines.base import BaseDataPreparer, write_sharegpt


@register_baseline("ours")
class OursPreparer(BaseDataPreparer):
    def prepare(self, output_dir: Path) -> Path:
        params = self.spec.params
        src: Path
        if params.get("round_data"):
            src = Path(params["round_data"])
        elif params.get("filtered_pool"):
            src = Path(params["filtered_pool"])
        elif "round" in params and self.spec.raw_pool_path:
            base = Path(self.spec.raw_pool_path)
            src = base / f"round_{params['round']}" / f"mixed_round_{params['round']}.json"
        else:
            raise ValueError(
                "ours baseline needs one of: params.round_data, params.filtered_pool, "
                "or (raw_pool_path + params.round)",
            )

        if not src.exists():
            raise FileNotFoundError(f"ours source file not found: {src}")

        if src.suffix == ".jsonl":
            from experiments.baselines.base import load_jsonl_pool
            data = load_jsonl_pool(src)
        else:
            data = json.loads(src.read_text("utf-8"))
        if self.spec.target_size and len(data) > self.spec.target_size:
            data = data[: self.spec.target_size]
        return write_sharegpt(data, Path(output_dir) / f"train_{self.spec.name}.json")
