"""GQA evaluator.

GQA questions have a single ground-truth short answer. Accuracy = exact match
after normalisation.

Expected file: ``spec.data_path`` points at ``testdev_balanced_questions.json``
or any JSON dict keyed by question_id with fields ``imageId``, ``question``,
``answer``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.eval.benchmarks import register_benchmark
from experiments.eval.benchmarks.base import (
    BaseEvaluator,
    Example,
    first_line,
    normalise_answer,
)


@register_benchmark("gqa")
class GQAEvaluator(BaseEvaluator):
    default_metric = "accuracy"

    def load_examples(self) -> list[Example]:
        raw = json.loads(Path(self.spec.data_path).read_text("utf-8"))
        img_dir = Path(self.spec.image_dir)
        examples: list[Example] = []
        for qid, q in raw.items():
            examples.append(Example(
                example_id=str(qid),
                image_path=str(img_dir / f"{q['imageId']}.jpg"),
                question=q["question"],
                gold=q["answer"],
                category=q.get("types", {}).get("structural", ""),
            ))
        return examples

    def build_prompt(self, ex: Example) -> tuple[list[dict], list[str]]:
        return (
            [{"role": "user",
              "content": f"<image>{ex.question}\nAnswer the question using a single word or phrase."}],
            [ex.image_path],
        )

    def score(self, examples: list[Example], predictions: list[str]) -> dict[str, Any]:
        per_cat_correct: dict[str, int] = {}
        per_cat_total: dict[str, int] = {}
        correct = 0
        for ex, pred in zip(examples, predictions):
            ok = normalise_answer(first_line(pred)) == normalise_answer(str(ex.gold))
            correct += int(ok)
            cat = ex.category or "all"
            per_cat_correct[cat] = per_cat_correct.get(cat, 0) + int(ok)
            per_cat_total[cat] = per_cat_total.get(cat, 0) + 1
        score = correct / max(len(examples), 1)
        sub = {c: per_cat_correct[c] / per_cat_total[c] for c in per_cat_total}
        return {"score": score, "sub_scores": sub}
