"""Base class for benchmark evaluators.

A concrete evaluator supplies three things:

1. ``load_examples``  : read the benchmark annotation file into a list of
                        ``(example_id, image_path, question, gold)`` tuples.
2. ``build_prompt``   : turn an example into the (messages, images) pair that
                        the MLLM infer hook expects (mirrors LlamaFactory format).
3. ``score``          : compare a list of predictions against the gold answers
                        and return a headline metric plus optional sub-scores.

The orchestration (batching, JSON I/O, timing) is shared via ``evaluate``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from experiments.types import BenchmarkResult, BenchmarkSpec, save_jsonl


# ---------------------------------------------------------------------------
# Type alias for the MLLM inference hook.
# The hook receives one example's (messages, images) plus the model path
# and returns the raw text output. This matches the signature of
# ``tool/multimodal_infer.generate_multimodal`` when run over a single sample.
# ---------------------------------------------------------------------------
MLLMInferFn = Callable[[list[dict], list[str], str], str]


@dataclass
class Example:
    example_id: str
    image_path: str
    question: str
    gold: Any                        # str | list[str] | dict (MC choices) | bool
    category: str = ""               # e.g. POPE pope-random / pope-popular


class BaseEvaluator:
    """Template method for benchmark evaluators."""

    benchmark_name: str = ""         # set by @register_benchmark
    default_metric: str = "accuracy"

    def __init__(self, spec: BenchmarkSpec) -> None:
        self.spec = spec

    # --- to be overridden by subclasses -----------------------------------

    def load_examples(self) -> list[Example]:
        raise NotImplementedError

    def build_prompt(self, ex: Example) -> tuple[list[dict], list[str]]:
        """Default: single-turn with <image> placeholder."""
        return (
            [{"role": "user", "content": f"<image>{ex.question}"}],
            [ex.image_path],
        )

    def score(self, examples: list[Example], predictions: list[str]) -> dict[str, Any]:
        """Return ``{"score": float, "sub_scores": dict[str, float]}``."""
        raise NotImplementedError

    # --- shared orchestration ---------------------------------------------

    def evaluate(
        self,
        infer_fn: MLLMInferFn,
        model_path: str,
        method_name: str,
        output_dir: Path,
    ) -> BenchmarkResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        examples = self.load_examples()
        if self.spec.max_samples is not None:
            examples = examples[: self.spec.max_samples]

        t0 = time.time()
        predictions: list[str] = []
        pred_records: list[dict] = []
        for ex in examples:
            messages, images = self.build_prompt(ex)
            raw = infer_fn(messages, images, model_path)
            predictions.append(raw)
            pred_records.append({
                "example_id": ex.example_id,
                "question": ex.question,
                "gold": ex.gold,
                "prediction": raw,
                "category": ex.category,
            })
        wall = time.time() - t0

        pred_path = output_dir / f"predictions_{self.benchmark_name}.jsonl"
        save_jsonl(pred_path, pred_records)

        scored = self.score(examples, predictions)

        return BenchmarkResult(
            benchmark=self.benchmark_name,
            method=method_name,
            model_path=model_path,
            metric=self.spec.metric or self.default_metric,
            score=float(scored["score"]),
            num_samples=len(examples),
            sub_scores=scored.get("sub_scores", {}),
            predictions_path=str(pred_path),
            wall_time_sec=wall,
        )


# ---------------------------------------------------------------------------
# Common text-normalisation helpers reused by several benchmarks.
# ---------------------------------------------------------------------------

def normalise_answer(s: str) -> str:
    """VQA-style answer normalisation: lowercase, strip, collapse whitespace."""
    import re
    s = s.lower().strip()
    s = re.sub(r"[\.\,\!\?\"']", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def first_line(s: str) -> str:
    """Take the first non-empty line — MLLMs often emit CoT after a newline."""
    for line in s.splitlines():
        line = line.strip()
        if line:
            return line
    return s.strip()


def extract_choice_letter(s: str, choices: Iterable[str] = ("A", "B", "C", "D", "E", "F")) -> str | None:
    """Find the first A/B/C/D token in ``s``; returns None if absent."""
    import re
    choice_set = {c.upper() for c in choices}
    for tok in re.findall(r"\b([A-F])\b", s.upper()):
        if tok in choice_set:
            return tok
    return None
