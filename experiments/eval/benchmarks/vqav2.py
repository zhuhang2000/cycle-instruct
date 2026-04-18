"""VQAv2 evaluator.

Expected layout of ``spec.data_path``:
    questions JSON : {"questions": [{"question_id": int, "image_id": int, "question": str}, ...]}
    annotations  JSON (gold): {"annotations": [{"question_id": int, "answers": [{"answer": str}, ...]}, ...]}

If only a merged annotation file is provided, pass it via ``spec.extras["merged_path"]``
and we'll read it directly.

Scoring follows the standard VQAv2 soft-accuracy:
    score = min(#humans_that_said_answer / 3, 1.0)
Averaged over all questions.
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


@register_benchmark("vqav2")
class VQAv2Evaluator(BaseEvaluator):
    default_metric = "accuracy"

    def load_examples(self) -> list[Example]:
        root = Path(self.spec.data_path)
        q_file = root / f"v2_OpenEnded_mscoco_{self.spec.split}_questions.json"
        a_file = root / f"v2_mscoco_{self.spec.split}_annotations.json"
        if "merged_path" in self.spec.extras:
            merged = json.loads(Path(self.spec.extras["merged_path"]).read_text("utf-8"))
            questions = merged["questions"]
            annotations = {a["question_id"]: a for a in merged["annotations"]}
        else:
            questions = json.loads(q_file.read_text("utf-8"))["questions"]
            annotations = {
                a["question_id"]: a
                for a in json.loads(a_file.read_text("utf-8"))["annotations"]
            }

        img_dir = Path(self.spec.image_dir)
        examples: list[Example] = []
        for q in questions:
            qid = q["question_id"]
            ann = annotations.get(qid)
            if ann is None:
                continue
            img_name = f"COCO_{self.spec.split}_{q['image_id']:012d}.jpg"
            examples.append(Example(
                example_id=str(qid),
                image_path=str(img_dir / img_name),
                question=q["question"],
                gold=[a["answer"] for a in ann["answers"]],
            ))
        return examples

    def build_prompt(self, ex: Example) -> tuple[list[dict], list[str]]:
        return (
            [{"role": "user",
              "content": f"<image>{ex.question}\nAnswer the question using a single word or phrase."}],
            [ex.image_path],
        )

    def score(self, examples: list[Example], predictions: list[str]) -> dict[str, Any]:
        total = 0.0
        for ex, pred in zip(examples, predictions):
            p = normalise_answer(first_line(pred))
            golds = [normalise_answer(g) for g in ex.gold]
            matches = sum(1 for g in golds if g == p)
            total += min(matches / 3.0, 1.0)
        score = total / max(len(examples), 1)
        return {"score": score, "sub_scores": {}}
