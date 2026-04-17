"""Iterative self-training package for Cycle-Instruct.

This package provides the orchestration layer that repeatedly:
1. generates VQA pairs with the current MLLM,
2. verifies cycle consistency and filters,
3. mixes the new samples with seed + historical high-quality samples,
4. fine-tunes a fresh LoRA adapter from the base model,
5. merges it, and uses the merged model as next round's generator.

Key invariant (see README): every round's LoRA is initialised from the
original ``base_model_path``. The merged model is only used as the *data
generator* for the next round, never as the next round's training start.
"""

from code.iterative.metrics import RoundMetrics, save_metrics, load_all_rounds, should_stop
from code.iterative.round_config import RoundTrainingConfig, get_training_config

__all__ = [
    "RoundMetrics",
    "save_metrics",
    "load_all_rounds",
    "should_stop",
    "RoundTrainingConfig",
    "get_training_config",
]
