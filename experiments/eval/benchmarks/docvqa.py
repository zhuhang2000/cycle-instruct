"""DocVQA evaluator — ANLS (Average Normalized Levenshtein Similarity).

Expected file layout
--------------------
``spec.data_path`` points at DocVQA ``val_v1.0.json`` (or similar). The schema
used here matches the public release:
    {"data": [{"questionId": int, "question": str, "image": "<path>",
                "answers": ["..", ..]}, ...]}

ANLS per question is defined in Biten et al. (2019): take the best normalised
Levenshtein distance between prediction and any gold answer, threshold at
``tau`` (0.5 by default — below threshold the score is 0).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.eval.benchmarks import register_benchmark
from experiments.eval.benchmarks.base import BaseEvaluator, Example, first_line


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _nls(pred: str, gold: str) -> float:
    p = pred.strip().lower()
    g = gold.strip().lower()
    if not p and not g:
        return 1.0
    denom = max(len(p), len(g), 1)
    return 1.0 - _levenshtein(p, g) / denom


def anls(pred: str, golds: list[str], tau: float = 0.5) -> float:
    if not golds:
        return 0.0
    best = max(_nls(pred, g) for g in golds)
    return best if best >= tau else 0.0


@register_benchmark("docvqa")
class DocVQAEvaluator(BaseEvaluator):
    default_metric = "anls"

    def load_examples(self) -> list[Example]:
        raw = json.loads(Path(self.spec.data_path).read_text("utf-8"))
        items = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        img_dir = Path(self.spec.image_dir)
        examples: list[Example] = []
        for item in items:
            img_name = item.get("image") or item.get("image_file") or ""
            img_path = str(img_dir / img_name) if img_name else ""
            examples.append(Example(
                example_id=str(item.get("questionId", item.get("question_id", ""))),
                image_path=img_path,
                question=item["question"],
                gold=item.get("answers", []),
                category=item.get("doc_type", ""),
            ))
        return examples

    def build_prompt(self, ex: Example) -> tuple[list[dict], list[str]]:
        return (
            [{"role": "user",
              "content": f"<image>{ex.question}\nAnswer the question using the text shown in the document."}],
            [ex.image_path],
        )

    def score(self, examples: list[Example], predictions: list[str]) -> dict[str, Any]:
        tau = float(self.spec.extras.get("anls_tau", 0.5))
        total = 0.0
        per_cat_sum: dict[str, float] = {}
        per_cat_n: dict[str, int] = {}
        for ex, pred in zip(examples, predictions):
            golds = ex.gold if isinstance(ex.gold, list) else [str(ex.gold)]
            s = anls(first_line(pred), [str(g) for g in golds], tau=tau)
            total += s
            cat = ex.category or "all"
            per_cat_sum[cat] = per_cat_sum.get(cat, 0.0) + s
            per_cat_n[cat] = per_cat_n.get(cat, 0) + 1
        n = max(len(examples), 1)
        sub = {c: per_cat_sum[c] / per_cat_n[c] for c in per_cat_n}
        return {"score": total / n, "sub_scores": sub}
