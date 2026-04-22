"""Tests for diversity metrics: distinct-n / Self-BLEU / MTLD edges."""
from __future__ import annotations

from experiments.intrinsic.diversity import (
    DiversityMetric,
    distinct_n,
    mtld,
    self_bleu,
    type_token_ratio,
)


def test_distinct_n_all_same_returns_small_fraction():
    # One repeated sentence of 5 tokens -> 4 bigrams, all identical
    texts = ["the cat sat on mat"] * 10
    d2 = distinct_n(texts, 2)
    # 40 bigrams total, 4 unique
    assert abs(d2 - 4 / 40) < 1e-9


def test_distinct_n_all_unique_is_one():
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    assert distinct_n(texts, 2) == 1.0


def test_distinct_n_empty_returns_zero():
    assert distinct_n([], 2) == 0.0
    assert distinct_n(["a"], 2) == 0.0


def test_self_bleu_identical_corpus_near_one():
    texts = ["the quick brown fox jumps over the lazy dog"] * 5
    score = self_bleu(texts, n_sample=10, seed=0)
    assert score > 0.9


def test_self_bleu_fully_distinct_low():
    texts = [
        "alpha beta gamma delta epsilon",
        "one two three four five",
        "foo bar baz qux quux",
        "red green blue yellow purple",
    ]
    score = self_bleu(texts, n_sample=20, seed=0)
    assert score < 0.2


def test_mtld_monotonic_in_repetition():
    diverse = " ".join(f"word{i}" for i in range(60))
    repeated = " ".join(["word"] * 60)
    assert mtld(diverse) > mtld(repeated)


def test_ttr_edges():
    assert type_token_ratio([""]) == 0.0
    assert type_token_ratio(["a b c"]) == 1.0
    assert type_token_ratio(["a a a"]) == 1 / 3


def test_diversity_metric_end_to_end():
    samples = [
        {"question": f"question {i}", "answer": f"answer {i}"}
        for i in range(5)
    ]
    result = DiversityMetric().compute(samples, n_sample_self_bleu=10)
    assert result["num_samples"] == 5
    assert 0 <= result["distinct_2_q"] <= 1
    assert result["self_bleu_4_q"] >= 0
    assert "mtld_mean" in result
