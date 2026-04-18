"""MMBench evaluator (multiple-choice, 20 L2 skill categories).

Expected TSV columns (MMBench-dev): ``index``, ``question``, ``hint``,
``A``, ``B``, ``C``, ``D``, ``answer`` (A/B/C/D letter), ``category``, ``image`` (base64).

For scaffolding we assume images are pre-extracted to ``spec.image_dir`` and
the TSV is at ``spec.data_path``. Sub-score keyed by ``l2-category``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from experiments.eval.benchmarks import register_benchmark
from experiments.eval.benchmarks.base import (
    BaseEvaluator,
    Example,
    extract_choice_letter,
)


@register_benchmark("mmbench")
class MMBenchEvaluator(BaseEvaluator):
    default_metric = "mc_acc"

    def load_examples(self) -> list[Example]:
        img_dir = Path(self.spec.image_dir)
        examples: list[Example] = []
        with Path(self.spec.data_path).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                idx = row["index"]
                choices = {k: row[k] for k in ("A", "B", "C", "D") if row.get(k)}
                examples.append(Example(
                    example_id=str(idx),
                    image_path=str(img_dir / f"{idx}.jpg"),
                    question=row["question"],
                    gold={"letter": row.get("answer", "").strip().upper(), "choices": choices,
                          "hint": row.get("hint", "")},
                    category=row.get("l2-category", row.get("category", "")),
                ))
        return examples

    def build_prompt(self, ex: Example) -> tuple[list[dict], list[str]]:
        gold: dict = ex.gold  # type: ignore[assignment]
        choices_text = "\n".join(f"{k}. {v}" for k, v in gold["choices"].items())
        hint = gold.get("hint", "")
        hint_block = f"Hint: {hint}\n" if hint else ""
        prompt = (
            f"<image>{hint_block}Question: {ex.question}\n"
            f"Options:\n{choices_text}\n"
            "Answer with the option letter only."
        )
        return [{"role": "user", "content": prompt}], [ex.image_path]

    def score(self, examples: list[Example], predictions: list[str]) -> dict[str, Any]:
        per_cat_c: dict[str, int] = {}
        per_cat_t: dict[str, int] = {}
        correct = 0
        for ex, pred in zip(examples, predictions):
            gold_letter = ex.gold["letter"]      # type: ignore[index]
            got = extract_choice_letter(pred, choices=list(ex.gold["choices"].keys()))  # type: ignore[index]
            ok = (got == gold_letter)
            correct += int(ok)
            cat = ex.category or "all"
            per_cat_c[cat] = per_cat_c.get(cat, 0) + int(ok)
            per_cat_t[cat] = per_cat_t.get(cat, 0) + 1
        score = correct / max(len(examples), 1)
        sub = {c: per_cat_c[c] / per_cat_t[c] for c in per_cat_t}
        return {"score": score, "sub_scores": sub}
