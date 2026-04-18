"""HallusionBench evaluator.

HallusionBench tests visual hallucinations via paired (image, edited-image)
yes/no questions. Three headline metrics:

- ``aAcc`` — all-question accuracy (per-question)
- ``qAcc`` — question-pair accuracy (both halves of a pair correct)
- ``fAcc`` — figure accuracy (all questions about one figure correct)

Input JSON layout (public release simplified):
    [{"question_id": str, "figure_id": str, "question": str,
      "gt_answer_details": "Yes" | "No", "filename": str,
      "category": "VD" | "VS"}, ...]
We pair questions by ``figure_id`` + question text when the dataset has a
``set_id`` / ``question_pair_id`` field; otherwise ``qAcc`` defaults to ``aAcc``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.eval.benchmarks import register_benchmark
from experiments.eval.benchmarks.base import BaseEvaluator, Example, first_line


def _yes_no(text: str) -> str | None:
    t = first_line(text).strip().lower()
    while t and t[-1] in ".!?,":
        t = t[:-1]
    if t in ("yes", "yeah", "yep"):
        return "yes"
    if t in ("no", "nope"):
        return "no"
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return None


@register_benchmark("hallusion")
class HallusionEvaluator(BaseEvaluator):
    default_metric = "accuracy"

    def load_examples(self) -> list[Example]:
        raw = json.loads(Path(self.spec.data_path).read_text("utf-8"))
        items = raw if isinstance(raw, list) else raw.get("data", [])
        img_dir = Path(self.spec.image_dir)
        examples: list[Example] = []
        for item in items:
            gold = str(item.get("gt_answer_details", item.get("gt_answer", ""))).strip().lower()
            if gold in ("1", "true"):
                gold = "yes"
            elif gold in ("0", "false"):
                gold = "no"
            img_name = item.get("filename") or item.get("image") or ""
            img_path = str(img_dir / img_name) if img_name else ""
            qid = str(item.get("question_id", item.get("qid", len(examples))))
            examples.append(Example(
                example_id=qid,
                image_path=img_path,
                question=item["question"],
                gold={
                    "label": gold,
                    "figure_id": str(item.get("figure_id", "")),
                    "set_id": str(item.get("set_id", item.get("question_pair_id", qid))),
                },
                category=item.get("category", ""),
            ))
        return examples

    def build_prompt(self, ex: Example) -> tuple[list[dict], list[str]]:
        return (
            [{"role": "user",
              "content": f"<image>{ex.question}\nAnswer with yes or no."}],
            [ex.image_path],
        )

    def score(self, examples: list[Example], predictions: list[str]) -> dict[str, Any]:
        per_q: list[bool] = []
        unparse = 0
        pair_groups: dict[str, list[bool]] = defaultdict(list)
        fig_groups: dict[str, list[bool]] = defaultdict(list)
        per_cat_c: dict[str, int] = {}
        per_cat_t: dict[str, int] = {}

        for ex, pred in zip(examples, predictions):
            got = _yes_no(pred)
            if got is None:
                unparse += 1
                got = "no"
            gold = ex.gold["label"]           # type: ignore[index]
            ok = (got == gold)
            per_q.append(ok)
            pair_groups[ex.gold["set_id"]].append(ok)          # type: ignore[index]
            fig_groups[ex.gold["figure_id"]].append(ok)        # type: ignore[index]
            cat = ex.category or "all"
            per_cat_c[cat] = per_cat_c.get(cat, 0) + int(ok)
            per_cat_t[cat] = per_cat_t.get(cat, 0) + 1

        n = max(len(per_q), 1)
        a_acc = sum(per_q) / n
        q_acc = (sum(1 for v in pair_groups.values() if all(v))
                 / max(len(pair_groups), 1))
        f_acc = (sum(1 for v in fig_groups.values() if all(v))
                 / max(len(fig_groups), 1))
        sub = {c: per_cat_c[c] / per_cat_t[c] for c in per_cat_t}
        sub.update({
            "aAcc": a_acc,
            "qAcc": q_acc,
            "fAcc": f_acc,
            "unparseable_ratio": unparse / n,
        })
        return {"score": a_acc, "sub_scores": sub}
