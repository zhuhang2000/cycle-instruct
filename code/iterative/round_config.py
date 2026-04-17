"""Per-round LoRA training hyper-parameters.

Implements the decay schedule described in the Week-1 plan:

==========  ====  ========  ==========  ============  ========
Round        LR    Epochs    LoRA rank   LoRA alpha    Warmup
==========  ====  ========  ==========  ============  ========
0           1e-4      3          8           16         0.03
1           8e-5      2          8           16         0.03
2           5e-5      2         16           32         0.05
3           3e-5      1         16           32         0.05
4+          1e-5      1         16           32         0.10
==========  ====  ========  ==========  ============  ========

Rationale:
  * LR decays because the data distribution narrows → we want finer steps.
  * rank grows at round 2 because by then the model has to represent a
    broader behavioural shift; the smaller rank was enough to bootstrap.
  * epochs shrink so later rounds do not over-fit a small, repetitive mix.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoundTrainingConfig:
    """Hyper-parameters for a single LoRA training run."""

    learning_rate: float
    num_epochs: int
    lora_rank: int
    lora_alpha: int
    warmup_ratio: float
    batch_size: int = 1
    grad_accumulation: int = 8
    cutoff_len: int = 2048
    lr_scheduler_type: str = "cosine"
    lora_dropout: float = 0.05

    def as_cli_overrides(self) -> dict[str, str]:
        """Return a dict compatible with LlamaFactory CLI flag names."""
        return {
            "learning_rate": str(self.learning_rate),
            "num_train_epochs": str(self.num_epochs),
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "warmup_ratio": str(self.warmup_ratio),
            "per_device_train_batch_size": str(self.batch_size),
            "gradient_accumulation_steps": str(self.grad_accumulation),
            "cutoff_len": str(self.cutoff_len),
            "lr_scheduler_type": self.lr_scheduler_type,
            "lora_dropout": str(self.lora_dropout),
        }


# Ordered from round 0 onward. ``get_training_config`` clamps to the last
# entry for rounds beyond the schedule.
_DEFAULT_SCHEDULE: list[RoundTrainingConfig] = [
    RoundTrainingConfig(  # Round 0 — bootstrap
        learning_rate=1e-4, num_epochs=3,
        lora_rank=8, lora_alpha=16, warmup_ratio=0.03,
    ),
    RoundTrainingConfig(  # Round 1 — slight cool-down
        learning_rate=8e-5, num_epochs=2,
        lora_rank=8, lora_alpha=16, warmup_ratio=0.03,
    ),
    RoundTrainingConfig(  # Round 2 — widen capacity
        learning_rate=5e-5, num_epochs=2,
        lora_rank=16, lora_alpha=32, warmup_ratio=0.05,
    ),
    RoundTrainingConfig(  # Round 3 — fine-tune
        learning_rate=3e-5, num_epochs=1,
        lora_rank=16, lora_alpha=32, warmup_ratio=0.05,
    ),
    RoundTrainingConfig(  # Round 4+ — steady-state
        learning_rate=1e-5, num_epochs=1,
        lora_rank=16, lora_alpha=32, warmup_ratio=0.10,
    ),
]


def get_training_config(
    round_id: int,
    schedule: list[RoundTrainingConfig] | None = None,
) -> RoundTrainingConfig:
    """Return the training config for ``round_id``.

    Rounds past the end of ``schedule`` clamp to its last entry. Negative
    ``round_id`` clamps to 0 and emits no warning; callers should ensure
    they always pass non-negative ids.
    """
    sched = schedule if schedule is not None else _DEFAULT_SCHEDULE
    if not sched:
        raise ValueError("schedule must contain at least one entry")
    idx = max(0, min(round_id, len(sched) - 1))
    return sched[idx]


def default_schedule() -> list[RoundTrainingConfig]:
    """Return a *copy* of the default schedule (safe to mutate)."""
    return list(_DEFAULT_SCHEDULE)
