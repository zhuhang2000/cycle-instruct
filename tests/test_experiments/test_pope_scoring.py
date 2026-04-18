"""POPE score boundaries: yes/no parsing, F1, yes_ratio, unparseable handling."""
from __future__ import annotations

from experiments.eval.benchmarks.base import Example
from experiments.eval.benchmarks.pope import PopeEvaluator, _yes_no
from experiments.types import BenchmarkSpec


def _mk_evaluator() -> PopeEvaluator:
    return PopeEvaluator(BenchmarkSpec(name="pope"))


def test_yes_no_parse_variants() -> None:
    assert _yes_no("Yes.") == "yes"
    assert _yes_no("No, it is not.") == "no"
    assert _yes_no("YES") == "yes"
    assert _yes_no("yeah") == "yes"
    assert _yes_no("nope") == "no"
    assert _yes_no("maybe") is None
    assert _yes_no("") is None


def test_all_yes_high_recall_low_precision() -> None:
    ev = _mk_evaluator()
    exs = [Example(str(i), "", "q?", gold=("yes" if i % 2 == 0 else "no"))
           for i in range(10)]
    preds = ["yes"] * 10
    out = ev.score(exs, preds)
    assert 0.49 <= out["score"] <= 0.51
    sub = out["sub_scores"]
    assert sub["recall"] == 1.0
    assert abs(sub["yes_ratio"] - 1.0) < 1e-9


def test_all_no_inverse() -> None:
    ev = _mk_evaluator()
    exs = [Example(str(i), "", "q?", gold=("yes" if i % 2 == 0 else "no"))
           for i in range(10)]
    preds = ["no"] * 10
    out = ev.score(exs, preds)
    assert 0.49 <= out["score"] <= 0.51
    assert out["sub_scores"]["recall"] == 0.0
    assert out["sub_scores"]["yes_ratio"] == 0.0


def test_unparseable_counts_as_no() -> None:
    ev = _mk_evaluator()
    exs = [Example("1", "", "q?", gold="yes"), Example("2", "", "q?", gold="no")]
    preds = ["banana", "banana"]
    out = ev.score(exs, preds)
    assert out["sub_scores"]["unparseable_ratio"] == 1.0
    assert out["score"] == 0.5  # one "no" correct, one wrong
