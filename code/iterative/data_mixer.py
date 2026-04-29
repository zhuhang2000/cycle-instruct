"""Round-wise training-data mixing + historical-pool management.

Responsibilities
----------------
* Given this round's freshly filtered VQA pairs, mix them with the
  original seed set and a rolling "historical best" pool.
* Maintain the historical pool as a bounded JSONL file, keyed on
  ``image_id + question_hash`` and truncated to the top ``K`` by
  ``cycle_score``.
* Emit ShareGPT JSON + a ``dataset_info.json`` fragment that LlamaFactory
  can consume directly.

Streaming design: both ``sample_from_jsonl`` and
``update_historical_pool`` read/write line-by-line to keep memory usage
constant even when the pool grows to tens of thousands of entries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keying & deduplication
# ---------------------------------------------------------------------------


def _normalise_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _sample_key(sample: dict[str, Any]) -> str:
    """Return a stable deduplication key for one ShareGPT sample."""
    images = sample.get("images") or []
    image_id = images[0] if images else sample.get("image_id", "")
    q = ""
    for m in sample.get("messages", []) or []:
        if m.get("role") == "user":
            q = m.get("content", "")
            break
    if not q and "question" in sample:
        q = sample["question"]
    q_norm = _normalise_question(re.sub(r"<image>\s*", "", q))
    digest = hashlib.md5(f"{image_id}||{q_norm}".encode("utf-8")).hexdigest()
    return digest


def deduplicate_by_key(
    samples: Iterable[dict[str, Any]],
    key: Callable[[dict[str, Any]], str] = _sample_key,
) -> list[dict[str, Any]]:
    """Keep the first occurrence of each ``key(sample)``."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for s in samples:
        k = key(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Reservoir sampling from JSONL / JSON
# ---------------------------------------------------------------------------


def _iter_samples(path: Path) -> Iterable[dict[str, Any]]:
    """Yield samples from either a JSONL file or a top-level JSON array."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)
        if first == "[":
            # top-level JSON array
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            return
        # else JSONL (one object per line)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("[data_mixer] skipping malformed line in %s", path)


def sample_from_jsonl(
    path: Path,
    n: int,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Reservoir-sample ``n`` items from ``path`` (JSONL or JSON array).

    Deterministic given ``seed``. Returns an empty list if ``path`` does
    not exist or is empty. Returns all items (unshuffled) if the file
    contains ``<= n`` items.
    """
    if n <= 0:
        return []
    path = Path(path)
    if not path.is_file():
        return []

    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    for i, item in enumerate(_iter_samples(path)):
        if i < n:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = item
    return reservoir


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------


def _resolve_ratio(schedule: list[float], round_id: int) -> float:
    if not schedule:
        return 0.0
    return schedule[min(max(0, round_id), len(schedule) - 1)]


def mix_training_data(
    new_data: list[dict[str, Any]],
    *,
    round_id: int,
    target_total: int,
    initial_data_path: Path | str,
    historical_pool_path: Path | str | None,
    new_ratio_schedule: list[float],
    original_ratio_schedule: list[float],
    historical_ratio_schedule: list[float],
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample + dedup + shuffle a mixture for round ``round_id``.

    Any pool returning fewer items than requested is simply undersampled;
    the resulting mixture may be smaller than ``target_total`` — callers
    that need a hard size can pad by re-sampling the largest remaining
    pool, but for SFT that is rarely necessary.
    """
    if seed is None:
        seed = round_id
    rng = random.Random(seed)

    r_new = _resolve_ratio(new_ratio_schedule, round_id)
    r_orig = _resolve_ratio(original_ratio_schedule, round_id)
    r_hist = _resolve_ratio(historical_ratio_schedule, round_id)

    n_new = int(target_total * r_new)
    n_orig = int(target_total * r_orig)
    n_hist = int(target_total * r_hist)

    mixed: list[dict[str, Any]] = []

    # New data — sample from the in-memory list
    if n_new > 0 and new_data:
        mixed.extend(rng.sample(new_data, min(n_new, len(new_data))))

    # Original seed
    if n_orig > 0:
        mixed.extend(sample_from_jsonl(Path(initial_data_path), n_orig, seed=seed))

    # Historical pool
    if n_hist > 0 and historical_pool_path:
        hist_path = Path(historical_pool_path)
        if hist_path.is_file():
            mixed.extend(sample_from_jsonl(hist_path, n_hist, seed=seed + 1))

    # Dedup + shuffle
    mixed = deduplicate_by_key(mixed)
    rng.shuffle(mixed)

    logger.info(
        "[data_mixer] round=%d target=%d ratios=(new=%.2f, orig=%.2f, hist=%.2f) "
        "requested=(%d, %d, %d) final=%d",
        round_id, target_total, r_new, r_orig, r_hist,
        n_new, n_orig, n_hist, len(mixed),
    )
    return mixed


# ---------------------------------------------------------------------------
# Historical pool
# ---------------------------------------------------------------------------


def update_historical_pool(
    new_data: list[dict[str, Any]],
    *,
    historical_pool_path: Path | str,
    pool_size: int = 5000,
    quality_threshold: float = 0.85,
    round_id: int = 0,
    score_key: str = "cycle_score",
) -> int:
    """Merge high-quality ``new_data`` into the rolling pool.

    Writes a single rewrite of ``historical_pool_path`` (JSONL). Returns
    the final pool size. Samples lacking ``cycle_scores`` are skipped —
    the pool is meant to hold *verified* samples only.
    """
    path = Path(historical_pool_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load existing pool (small-ish by design: at most ``pool_size`` items)
    existing: list[dict[str, Any]] = list(_iter_samples(path))

    # 2. Annotate + filter new candidates
    candidates: list[dict[str, Any]] = []
    for s in new_data:
        cs = (s.get("cycle_scores") or {}).get("composite")
        if cs is None:
            cs = s.get(score_key)
        if cs is None or cs < quality_threshold:
            continue
        enriched = dict(s)
        enriched.setdefault("round_added", round_id)
        enriched[score_key] = float(cs)
        candidates.append(enriched)

    # 3. Merge + dedup (existing items win on ties — preserves earlier round_added)
    merged_dict: dict[str, dict[str, Any]] = {}
    for s in existing + candidates:
        key = _sample_key(s)
        incumbent = merged_dict.get(key)
        if incumbent is None:
            merged_dict[key] = s
            continue
        if float(s.get(score_key, 0.0)) > float(incumbent.get(score_key, 0.0)):
            merged_dict[key] = s
    merged = list(merged_dict.values())

    # 4. Cap by descending score
    merged.sort(key=lambda x: x.get(score_key, 0.0), reverse=True)
    merged = merged[:pool_size]

    # 5. Rewrite JSONL
    with path.open("w", encoding="utf-8") as f:
        for s in merged:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")

    logger.info(
        "[data_mixer] historical pool updated: %d existing + %d candidates → %d kept",
        len(existing), len(candidates), len(merged),
    )
    return len(merged)


# ---------------------------------------------------------------------------
# LlamaFactory dataset glue
# ---------------------------------------------------------------------------


def to_llamafactory_dataset(
    mixed_data: list[dict[str, Any]],
    *,
    output_dir: Path | str,
    dataset_name: str,
    dataset_info_path: Path | str | None = None,
) -> tuple[Path, Path | None]:
    """Dump mixed data to ``output_dir/<dataset_name>.json`` (ShareGPT).

    If ``dataset_info_path`` is supplied, its JSON is updated in-place
    (creating the file if missing) to register ``dataset_name`` so the
    training script can reference it by name.

    Returns ``(dataset_file, dataset_info_path | None)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = output_dir / f"{dataset_name}.json"

    with data_file.open("w", encoding="utf-8") as f:
        json.dump(mixed_data, f, ensure_ascii=False, indent=2)

    info_path: Path | None = None
    if dataset_info_path is not None:
        info_path = Path(dataset_info_path)
        info: dict[str, Any] = {}
        if info_path.is_file():
            try:
                with info_path.open("r", encoding="utf-8") as f:
                    info = json.load(f) or {}
            except json.JSONDecodeError:
                logger.warning(
                    "[data_mixer] %s was malformed; overwriting", info_path,
                )
                info = {}
        file_name = (
            data_file.name
            if data_file.parent.resolve() == info_path.parent.resolve()
            else str(data_file.resolve())
        )
        info[dataset_name] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
        info_path.parent.mkdir(parents=True, exist_ok=True)
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    logger.info(
        "[data_mixer] wrote dataset %s (%d samples) → %s",
        dataset_name, len(mixed_data), data_file,
    )
    return data_file, info_path
