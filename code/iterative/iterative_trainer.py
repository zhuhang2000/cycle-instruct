"""Iterative self-training controller for multimodal Cycle-Instruct.

Responsibilities
----------------
1. For each round 0..max_rounds-1:
    a. Pick generator model (base for round 0, previous round's merged
       model thereafter).
    b. Run Stage 1-3 (generate → verify → filter) to get round's VQAs.
    c. Mix with seed + historical pool (``data_mixer``).
    d. Down-sample over-represented QA types (``qa_templates``).
    e. Resolve per-round LoRA hyper-parameters (``round_config``).
    f. Launch fresh LoRA training **from base_model_path**.
    g. Merge LoRA → merged_model.
    h. Record ``RoundMetrics`` + maybe early-stop.
    i. Update the historical pool.

The I/O-heavy steps (generation / training / merging) are invoked via
thin hook callables. Tests (``tests/test_iterative/test_iterative_smoke.py``)
patch those hooks to run end-to-end without real models.

Run as a script:

    python -m code.iterative.iterative_trainer \\
        --base_model_path /path/to/Qwen3-VL-4B \\
        --initial_data_path seeds.json \\
        --raw_image_dir images/ \\
        --output_root runs/smoke/ \\
        --max_rounds 2 \\
        --samples_per_round 50

Add ``--dry-run`` to print the plan without invoking models or training.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from code.iterative.data_mixer import (
    mix_training_data,
    to_llamafactory_dataset,
    update_historical_pool,
)
from code.iterative.metrics import RoundMetrics, save_metrics, should_stop
from code.iterative.qa_templates import (
    compute_diversity_score,
    compute_type_distribution,
    rebalance_qa_types,
)
from code.iterative.round_config import (
    RoundTrainingConfig,
    get_training_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class IterativeConfig:
    """Top-level configuration for the iterative loop."""

    # ---- paths -----------------------------------------------------------
    base_model_path: str
    initial_data_path: str
    raw_image_dir: str
    output_root: str
    llamafactory_data_dir: str | None = None  # where to write dataset_info.json

    # ---- schedule --------------------------------------------------------
    max_rounds: int = 5
    samples_per_round: int = 2000

    # ---- mixing ratios (per round, clamp to last) -----------------------
    new_ratio_schedule: list[float] = field(
        default_factory=lambda: [0.0, 0.60, 0.50, 0.40, 0.30]
    )
    original_ratio_schedule: list[float] = field(
        default_factory=lambda: [1.0, 0.30, 0.30, 0.30, 0.30]
    )
    historical_ratio_schedule: list[float] = field(
        default_factory=lambda: [0.0, 0.10, 0.20, 0.30, 0.40]
    )

    # ---- historical pool ------------------------------------------------
    historical_pool_size: int = 5000
    historical_quality_threshold: float = 0.85

    # ---- QA diversity ---------------------------------------------------
    qa_type_min_fraction: float = 0.10
    qa_type_max_fraction: float = 0.35

    # ---- early-stop -----------------------------------------------------
    diversity_threshold: float = 0.60
    pass_rate_drop_threshold: float = 0.15
    drift_converged_threshold: float = 0.02
    patience: int = 2

    # ---- dry run --------------------------------------------------------
    dry_run: bool = False

    # -- derived helpers ---------------------------------------------------
    def round_dir(self, round_id: int) -> Path:
        return Path(self.output_root) / f"round_{round_id}"

    def historical_pool_path(self) -> Path:
        return Path(self.output_root) / "historical_pool.jsonl"


# ---------------------------------------------------------------------------
# Hooks (separated to make mocking trivial)
# ---------------------------------------------------------------------------


GenAndFilterFn = Callable[[str, str, Path, int], list[dict[str, Any]]]
LoraTrainFn = Callable[[str, Path, RoundTrainingConfig, Path, str], Path]
MergeFn = Callable[[str, Path, Path], None]


def run_generation_and_filter(
    generator_model_path: str,
    raw_image_dir: str,
    round_dir: Path,
    samples_to_generate: int,
) -> list[dict[str, Any]]:
    """Default Stage 1-3 driver.

    This is intentionally a light wrapper over the existing
    ``code/I2QA/*`` scripts. Unit tests patch this symbol; production
    runs let it shell-out to the same scripts that
    ``bash/run_multimodal_cycle.sh`` already invokes.

    Implementation note: this calls the I2QA scripts via subprocess to
    keep vLLM's CUDA contexts isolated per stage (vLLM does not release
    GPU memory on ``del``). Set ``CI_SKIP_HEAVY=1`` in tests.
    """
    round_dir.mkdir(parents=True, exist_ok=True)
    filtered_file = round_dir / "filtered_vqa.json"

    if os.environ.get("CI_SKIP_HEAVY"):
        logger.warning("[controller] CI_SKIP_HEAVY=1 — returning empty VQA list")
        filtered_file.write_text("[]", encoding="utf-8")
        return []

    cmd = [
        sys.executable, "-m", "code.I2QA.generate_vqa_pairs",
        "--model_path", generator_model_path,
        "--image_dir", raw_image_dir,
        "--output", str(round_dir / "raw_vqa.json"),
        "--num_samples", str(samples_to_generate),
    ]
    logger.info("[controller] generation: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Verify + filter (re-using existing scripts)
    subprocess.run([
        sys.executable, "-m", "code.I2QA.verify_cycle_consistency",
        "--input", str(round_dir / "raw_vqa.json"),
        "--output", str(round_dir / "scored_vqa.json"),
    ], check=True)

    subprocess.run([
        sys.executable, "-m", "code.I2QA.filter_and_export",
        "--input", str(round_dir / "scored_vqa.json"),
        "--output", str(filtered_file),
    ], check=True)

    with filtered_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_lora_training(
    base_model_path: str,
    dataset_file: Path,
    train_cfg: RoundTrainingConfig,
    round_dir: Path,
    dataset_name: str,
) -> Path:
    """Invoke LlamaFactory CLI to train a fresh LoRA adapter.

    Writes the adapter to ``round_dir/lora`` and returns that path.

    This mirrors ``code/A2Q/PEFT_LoRA.bash`` — production callers may
    prefer to template a shell script instead of calling ``llamafactory``
    directly from Python; both behaviours are trivially swappable.
    """
    lora_dir = round_dir / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("CI_SKIP_HEAVY"):
        logger.warning("[controller] CI_SKIP_HEAVY=1 — skipping real training")
        (lora_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        return lora_dir

    dataset_dir = dataset_file.parent
    cmd = [
        sys.executable, "-m", "llamafactory.cli", "train",
        "--stage", "sft",
        "--do_train", "True",
        "--model_name_or_path", base_model_path,
        "--dataset_dir", str(dataset_dir),
        "--dataset", dataset_name,
        "--output_dir", str(lora_dir),
        "--template", "qwen3",  # caller may override; qwen3 matches the base model
        "--finetuning_type", "lora",
        "--lora_target", "all",
        "--bf16", "True",
        "--plot_loss", "True",
        "--save_steps", "200",
        "--logging_steps", "20",
    ]
    for k, v in train_cfg.as_cli_overrides().items():
        cmd += [f"--{k}", v]
    logger.info("[controller] training: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return lora_dir


def run_merge_lora(
    base_model_path: str,
    lora_dir: Path,
    merged_dir: Path,
) -> None:
    """Merge the adapter into a standalone model via ``llamafactory-cli export``."""
    merged_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("CI_SKIP_HEAVY"):
        logger.warning("[controller] CI_SKIP_HEAVY=1 — skipping real merge")
        (merged_dir / "config.json").write_text("{}", encoding="utf-8")
        return

    cmd = [
        "llamafactory-cli", "export",
        "--model_name_or_path", base_model_path,
        "--adapter_name_or_path", str(lora_dir),
        "--template", "qwen3",
        "--finetuning_type", "lora",
        "--export_dir", str(merged_dir),
        "--export_size", "2",
        "--export_legacy_format", "False",
    ]
    logger.info("[controller] merging: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _summarise_cycle_scores(vqa_pairs: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mean/std over cycle score components."""
    import statistics as stats

    def _vals(key: str) -> list[float]:
        out: list[float] = []
        for s in vqa_pairs:
            cs = s.get("cycle_scores") or {}
            v = cs.get(key)
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    composite = _vals("composite")
    return {
        "mean_cycle_score": sum(composite) / len(composite) if composite else 0.0,
        "std_cycle_score": stats.pstdev(composite) if len(composite) > 1 else 0.0,
        "mean_ar": sum(_vals("ar")) / max(1, len(_vals("ar"))),
        "mean_clip": sum(_vals("clip")) / max(1, len(_vals("clip"))),
        "mean_qr": sum(_vals("qr")) / max(1, len(_vals("qr"))),
        "mean_ppl": sum(_vals("ppl")) / max(1, len(_vals("ppl"))),
    }


def _compute_drift(
    current: list[dict[str, Any]],
    previous_dist: dict[str, float] | None,
) -> float | None:
    """L1 distance between current and previous type distributions (cheap proxy)."""
    if previous_dist is None:
        return None
    cur = compute_type_distribution(current)
    return sum(abs(cur.get(k, 0.0) - previous_dist.get(k, 0.0)) for k in cur) / 2.0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_iterative_training(
    cfg: IterativeConfig,
    *,
    gen_filter_fn: GenAndFilterFn | None = None,
    train_fn: LoraTrainFn | None = None,
    merge_fn: MergeFn | None = None,
) -> list[RoundMetrics]:
    """Run the full iterative loop.

    Any of the three hooks may be overridden — tests inject fakes, future
    callers may inject alternative training backends.
    """
    gen_filter_fn = gen_filter_fn or run_generation_and_filter
    train_fn = train_fn or run_lora_training
    merge_fn = merge_fn or run_merge_lora

    Path(cfg.output_root).mkdir(parents=True, exist_ok=True)
    history: list[RoundMetrics] = []
    prev_type_dist: dict[str, float] | None = None

    for round_id in range(cfg.max_rounds):
        round_dir = cfg.round_dir(round_id)
        round_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=" * 72)
        logger.info("[iterative] Round %d: start", round_id)

        # (1) choose generator
        if round_id == 0:
            gen_model = cfg.base_model_path
        else:
            gen_model = str(cfg.round_dir(round_id - 1) / "merged_model")
            if not Path(gen_model).is_dir():
                logger.error(
                    "[iterative] Round %d: previous merged model missing at %s — aborting",
                    round_id, gen_model,
                )
                break

        if cfg.dry_run:
            logger.info(
                "[iterative] DRY-RUN Round %d: generator=%s, samples=%d",
                round_id, gen_model, cfg.samples_per_round,
            )
            train_cfg = get_training_config(round_id)
            logger.info("[iterative] DRY-RUN train_cfg=%s", train_cfg)
            continue

        # (2) Stage 1-3
        filtered = gen_filter_fn(
            gen_model, cfg.raw_image_dir, round_dir, cfg.samples_per_round,
        )
        n_generated_path = round_dir / "raw_vqa.json"
        num_generated = cfg.samples_per_round
        if n_generated_path.is_file():
            try:
                with n_generated_path.open("r", encoding="utf-8") as f:
                    num_generated = len(json.load(f))
            except Exception:
                pass

        # (3) mix
        mixed = mix_training_data(
            filtered,
            round_id=round_id,
            target_total=cfg.samples_per_round,
            initial_data_path=cfg.initial_data_path,
            historical_pool_path=cfg.historical_pool_path(),
            new_ratio_schedule=cfg.new_ratio_schedule,
            original_ratio_schedule=cfg.original_ratio_schedule,
            historical_ratio_schedule=cfg.historical_ratio_schedule,
        )

        # (4) QA diversity rebalance
        mixed = rebalance_qa_types(
            mixed,
            min_fraction=cfg.qa_type_min_fraction,
            max_fraction=cfg.qa_type_max_fraction,
            seed=round_id,
        )

        # (5) Write dataset + register with LlamaFactory
        dataset_name = f"mixed_round_{round_id}"
        dataset_info = (
            Path(cfg.llamafactory_data_dir) / "dataset_info.json"
            if cfg.llamafactory_data_dir else None
        )
        dataset_file, _ = to_llamafactory_dataset(
            mixed,
            output_dir=round_dir,
            dataset_name=dataset_name,
            dataset_info_path=dataset_info,
        )

        # (6) training config
        train_cfg = get_training_config(round_id)

        # (7) TRAIN — always from base model
        lora_dir = train_fn(
            cfg.base_model_path, dataset_file, train_cfg, round_dir, dataset_name,
        )

        # (8) merge
        merged_dir = round_dir / "merged_model"
        merge_fn(cfg.base_model_path, lora_dir, merged_dir)

        # (9) metrics
        cur_dist = compute_type_distribution(mixed)
        score_summary = _summarise_cycle_scores(filtered)
        m = RoundMetrics(
            round_id=round_id,
            num_samples_generated=num_generated,
            num_samples_passed_filter=len(filtered),
            pass_rate=(len(filtered) / num_generated) if num_generated > 0 else 0.0,
            **score_summary,
            train_loss_initial=_read_train_loss(lora_dir, first=True),
            train_loss_final=_read_train_loss(lora_dir, first=False),
            lora_rank=train_cfg.lora_rank,
            learning_rate=train_cfg.learning_rate,
            num_epochs=train_cfg.num_epochs,
            data_diversity_score=compute_diversity_score(cur_dist),
            drift_from_prev=_compute_drift(mixed, prev_type_dist),
            extras={"type_distribution": cur_dist},
        )
        save_metrics(round_dir, m)
        history.append(m)
        prev_type_dist = cur_dist

        # (10) update historical pool
        update_historical_pool(
            filtered,
            historical_pool_path=cfg.historical_pool_path(),
            pool_size=cfg.historical_pool_size,
            quality_threshold=cfg.historical_quality_threshold,
            round_id=round_id,
        )

        # (11) early stop?
        stop, reason = should_stop(
            history,
            max_rounds=cfg.max_rounds,
            pass_rate_drop_threshold=cfg.pass_rate_drop_threshold,
            diversity_threshold=cfg.diversity_threshold,
            drift_converged_threshold=cfg.drift_converged_threshold,
            patience=cfg.patience,
        )
        logger.info("[iterative] Round %d: %s (stop=%s)", round_id, reason, stop)
        if stop:
            break

    return history


def _read_train_loss(lora_dir: Path, *, first: bool) -> float:
    """Best-effort read of training loss from trainer_log.jsonl."""
    tlog = Path(lora_dir) / "trainer_log.jsonl"
    if not tlog.is_file():
        return 0.0
    try:
        lines = tlog.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        losses = [r["loss"] for r in records if "loss" in r]
        if not losses:
            return 0.0
        return float(losses[0] if first else losses[-1])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iterative_trainer",
        description="Iterative self-training controller for Cycle-Instruct",
    )
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--initial_data_path", required=True)
    p.add_argument("--raw_image_dir", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--llamafactory_data_dir", default=None)
    p.add_argument("--max_rounds", type=int, default=5)
    p.add_argument("--samples_per_round", type=int, default=2000)
    p.add_argument("--historical_pool_size", type=int, default=5000)
    p.add_argument("--dry_run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_argparser().parse_args(argv)
    cfg = IterativeConfig(
        base_model_path=args.base_model_path,
        initial_data_path=args.initial_data_path,
        raw_image_dir=args.raw_image_dir,
        output_root=args.output_root,
        llamafactory_data_dir=args.llamafactory_data_dir,
        max_rounds=args.max_rounds,
        samples_per_round=args.samples_per_round,
        historical_pool_size=args.historical_pool_size,
        dry_run=args.dry_run,
    )
    logger.info("[iterative] config: %s", asdict(cfg))
    history = run_iterative_training(cfg)
    logger.info("[iterative] completed %d rounds", len(history))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
