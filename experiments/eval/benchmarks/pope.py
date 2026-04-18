"""POPE evaluator (object-hallucination, yes/no).

POPE has three subsets: random / popular / adversarial. Each line is:
    {"question_id": int, "image": str, "text": str, "label": "yes" | "no", "category": str}

We compute accuracy + yes/no precision, recall, F1. The "yes-ratio" is a useful
hallucination signal: a model that says "yes" too often is hallucinating.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.eval.benchmarks import register_benchmark
from experiments.eval.benchmarks.base import BaseEvaluator, Example, first_line


def _yes_no(text: str) -> str | None:
    t = first_line(text).strip().lower()
    # Strip trailing punctuation
    while t and t[-1] in ".!?,":
        t = t[:-1]
    if t in ("yes", "yeah", "yep", "correct"):
        return "yes"
    if t in ("no", "nope", "incorrect"):
        return "no"
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return None


@register_benchmark("pope")
class PopeEvaluator(BaseEvaluator):
    default_metric = "accuracy"

    def load_examples(self) -> list[Example]:
        img_dir = Path(self.spec.image_dir)
        path = Path(self.spec.data_path)
        examples: list[Example] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                examples.append(Example(
                    example_id=str(rec["question_id"]),
                    image_path=str(img_dir / rec["image"]),
                    question=rec["text"],
                    gold=rec["label"].lower(),
                    category=rec.get("category", ""),
                ))
        return examples

    def build_prompt(self, ex: Example) -> tuple[list[dict], list[str]]:
        return (
            [{"role": "user",
              "content": f"<image>{ex.question}\nPlease answer with yes or no."}],
            [ex.image_path],
        )

    def score(self, examples: list[Example], predictions: list[str]) -> dict[str, Any]:
        tp = fp = fn = tn = 0
        yes_count = 0
        unknown = 0
        per_cat_c: dict[str, int] = {}
        per_cat_t: dict[str, int] = {}
        for ex, pred in zip(examples, predictions):
            got = _yes_no(pred)
            if got is None:
                unknown += 1
                got = "no"                 # treat non-parseable as "no" by convention
            gold = ex.gold                 # type: ignore[assignment]
            if got == "yes":
                yes_count += 1
            ok = (got == gold)
            if gold == "yes" and got == "yes":
                tp += 1
            elif gold == "yes" and got == "no":
                fn += 1
            elif gold == "no" and got == "yes":
                fp += 1
            else:
                tn += 1
            cat = ex.category or "all"
            per_cat_c[cat] = per_cat_c.get(cat, 0) + int(ok)
            per_cat_t[cat] = per_cat_t.get(cat, 0) + 1

        n = max(len(examples), 1)
        acc = (tp + tn) / n
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        yes_ratio = yes_count / n
        sub = {c: per_cat_c[c] / per_cat_t[c] for c in per_cat_t}
        sub.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "yes_ratio": yes_ratio,
            "unparseable_ratio": unknown / n,
        })
        return {"score": acc, "sub_scores": sub}
