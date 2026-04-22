"""Linguistic-quality metrics: length, template repetition, answer shape.

Grammar checking (LanguageTool) is optional — if the ``language_tool_python``
dependency isn't available, the grammar-error-rate field is omitted.
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments.intrinsic._io import (
    extract_answer,
    extract_question,
    load_vqa_jsonl,
    save_json,
)
from experiments.intrinsic.base import IntrinsicMetric, register_metric


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _length_stats(texts: Iterable[str]) -> dict[str, float]:
    lens = [len(_tokens(t)) for t in texts]
    if not lens:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    mean = sum(lens) / len(lens)
    var = sum((x - mean) ** 2 for x in lens) / len(lens) if len(lens) > 1 else 0.0
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "min": float(min(lens)),
        "max": float(max(lens)),
        "count": len(lens),
    }


def template_repeat_rate(texts: Iterable[str], prefix_tokens: int = 3) -> float:
    """Fraction of texts whose first ``prefix_tokens`` tokens form a shared prefix.

    A text is counted toward the repeat pool if its prefix appears more
    than once across the corpus.
    """
    prefixes = []
    for t in texts:
        toks = _tokens(t.lower())
        if len(toks) >= prefix_tokens:
            prefixes.append(" ".join(toks[:prefix_tokens]))
    if not prefixes:
        return 0.0
    counts = Counter(prefixes)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(prefixes)


def yes_no_rate(answers: Iterable[str]) -> float:
    total = 0
    yn = 0
    for a in answers:
        t = a.strip().lower().rstrip(".!?")
        if not t:
            continue
        total += 1
        if t in {"yes", "no"}:
            yn += 1
    return yn / total if total else 0.0


def sentence_shape_rate(answers: Iterable[str]) -> float:
    """Fraction of answers that start with an uppercase and end with punctuation."""
    total = 0
    ok = 0
    for a in answers:
        s = a.strip()
        if not s:
            continue
        total += 1
        if s[0].isupper() and s[-1] in ".!?":
            ok += 1
    return ok / total if total else 0.0


def grammar_error_rate(texts: list[str], language: str = "en-US") -> float | None:
    try:
        import language_tool_python  # noqa: WPS433
    except ImportError:
        return None
    try:
        tool = language_tool_python.LanguageTool(language)
    except Exception:  # noqa: BLE001 - typical startup error when Java is absent.
        return None
    errors = 0
    total = 0
    for t in texts:
        if not t.strip():
            continue
        total += 1
        if tool.check(t):
            errors += 1
    return errors / total if total else 0.0


@register_metric("linguistic")
class LinguisticQualityMetric(IntrinsicMetric):
    """Module-5 metric: length, template, answer-shape, optional grammar."""

    def compute(
        self,
        samples: list[dict],
        *,
        language: str = "en-US",
        run_grammar_check: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        questions = [extract_question(s) for s in samples]
        answers = [extract_answer(s) for s in samples]

        result: dict[str, Any] = {
            "num_samples": len(samples),
            "question_length": _length_stats(questions),
            "answer_length": _length_stats(answers),
            "template_repeat_rate_q": template_repeat_rate(questions),
            "template_repeat_rate_a": template_repeat_rate(answers),
            "yes_no_answer_rate": yes_no_rate(answers),
            "sentence_shape_rate_a": sentence_shape_rate(answers),
        }
        if run_grammar_check:
            rate = grammar_error_rate(answers, language)
            if rate is not None:
                result["grammar_error_rate_a"] = rate
        return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Linguistic quality metrics.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--language", type=str, default="en-US")
    p.add_argument("--grammar-check", action="store_true")
    args = p.parse_args(argv)
    samples = load_vqa_jsonl(args.input)
    result = LinguisticQualityMetric().compute(
        samples, language=args.language, run_grammar_check=args.grammar_check,
    )
    if args.out:
        save_json(args.out, result)
    else:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
