"""Round-level metrics for iterative self-training.

Captures per-round statistics (data, cycle scores, training loss,
diversity, drift) and implements a multi-signal early-stop rule used by
``iterative_trainer.run_iterative_training``.

Persisted as ``round_<N>/metrics.json`` and reloaded on resume.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Statistics recorded at the end of one iterative round."""

    round_id: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ---- data statistics --------------------------------------------------
    num_samples_generated: int = 0
    num_samples_passed_filter: int = 0
    pass_rate: float = 0.0

    # ---- cycle-score summary ---------------------------------------------
    mean_cycle_score: float = 0.0
    std_cycle_score: float = 0.0
    mean_ar: float = 0.0
    mean_clip: float = 0.0
    mean_qr: float = 0.0
    mean_ppl: float = 0.0

    # ---- training statistics ---------------------------------------------
    train_loss_initial: float = 0.0
    train_loss_final: float = 0.0
    lora_rank: int = 0
    learning_rate: float = 0.0
    num_epochs: int = 0

    # ---- evaluation / convergence signals --------------------------------
    eval_accuracy: float | None = None
    data_diversity_score: float = 0.0
    drift_from_prev: float | None = None

    # ---- freeform extras --------------------------------------------------
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoundMetrics":
        known = {f for f in cls.__dataclass_fields__}
        extras = {k: v for k, v in data.items() if k not in known}
        filtered = {k: v for k, v in data.items() if k in known}
        filtered.setdefault("extras", {}).update(extras)
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_metrics(round_dir: Path, metrics: RoundMetrics) -> Path:
    """Write ``round_dir/metrics.json``. Creates ``round_dir`` if missing."""
    round_dir = Path(round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    out = round_dir / "metrics.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info("[metrics] saved round %d metrics → %s", metrics.round_id, out)
    return out


def load_all_rounds(base_dir: Path) -> list[RoundMetrics]:
    """Scan ``base_dir/round_*/metrics.json`` and return history sorted by round_id."""
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return []
    found: list[RoundMetrics] = []
    for sub in sorted(base_dir.glob("round_*")):
        mfile = sub / "metrics.json"
        if not mfile.is_file():
            continue
        try:
            with mfile.open("r", encoding="utf-8") as f:
                found.append(RoundMetrics.from_dict(json.load(f)))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[metrics] failed to load %s: %s", mfile, exc)
    found.sort(key=lambda m: m.round_id)
    return found


# ---------------------------------------------------------------------------
# Early-stop rule
# ---------------------------------------------------------------------------


def should_stop(
    history: list[RoundMetrics],
    *,
    max_rounds: int = 5,
    pass_rate_drop_threshold: float = 0.15,
    diversity_threshold: float = 0.6,
    drift_converged_threshold: float = 0.02,
    patience: int = 2,
) -> tuple[bool, str]:
    """Return ``(stop, reason)``.

    Fires on any of:

    1. ``pass_rate`` fell by more than ``pass_rate_drop_threshold`` for
       ``patience`` consecutive rounds (collapse detector).
    2. ``mean_cycle_score`` monotonically decreased for ``patience``
       consecutive rounds.
    3. ``data_diversity_score < diversity_threshold`` (type distribution
       collapse).
    4. ``drift_from_prev < drift_converged_threshold`` (converged — no more
       gains to be had).
    5. ``len(history) >= max_rounds``.

    Rules 1, 2 and 4 require a minimum of ``patience + 1`` rounds of
    history; otherwise they are ignored. This makes the function safe to
    call from round 0.
    """
    if not history:
        return False, "empty-history"

    latest = history[-1]

    # Rule 5: max rounds (round_id is 0-indexed)
    if latest.round_id >= max_rounds - 1 and len(history) >= max_rounds:
        return True, f"max-rounds-reached({max_rounds})"

    # Rule 3: diversity collapse — single-round signal
    if latest.data_diversity_score and latest.data_diversity_score < diversity_threshold:
        return True, (
            f"diversity-collapse({latest.data_diversity_score:.3f} < "
            f"{diversity_threshold})"
        )

    # Rule 4: drift converged
    if (
        latest.drift_from_prev is not None
        and latest.drift_from_prev < drift_converged_threshold
        and latest.round_id >= 1
    ):
        return True, f"drift-converged({latest.drift_from_prev:.4f})"

    # Need enough history for trend-based rules
    if len(history) < patience + 1:
        return False, "insufficient-history"

    tail = history[-(patience + 1) :]

    # Rule 1: pass-rate drop for ``patience`` consecutive rounds
    drops = [
        tail[i].pass_rate - tail[i + 1].pass_rate
        for i in range(len(tail) - 1)
    ]
    if all(d > pass_rate_drop_threshold for d in drops):
        return True, (
            f"pass-rate-collapse(drops={[round(d, 3) for d in drops]})"
        )

    # Rule 2: cycle-score monotonic decrease
    cycles = [m.mean_cycle_score for m in tail]
    if all(cycles[i] > cycles[i + 1] for i in range(len(cycles) - 1)):
        return True, (
            f"cycle-score-decreasing({[round(c, 3) for c in cycles]})"
        )

    return False, "healthy"
