"""Unit tests for code/iterative/qa_templates.py."""
from __future__ import annotations

import pytest

from code.iterative.qa_templates import (
    QA_TEMPLATES,
    QA_TYPES,
    build_typed_instruction,
    classify_qa_type,
    compute_diversity_score,
    compute_type_distribution,
    rebalance_qa_types,
)


def _sample(question: str, image: str = "img1.jpg") -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"<image>{question}"},
            {"role": "assistant", "content": "ok"},
        ],
        "images": [image],
    }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_all_six_templates_present() -> None:
    assert set(QA_TEMPLATES.keys()) == set(QA_TYPES)
    assert len(QA_TYPES) == 6


def test_build_typed_instruction_returns_strings() -> None:
    for t in QA_TYPES:
        sys_p, user_p = build_typed_instruction(t)
        assert isinstance(sys_p, str) and sys_p
        assert isinstance(user_p, str) and user_p


def test_build_typed_instruction_unknown_raises() -> None:
    with pytest.raises(KeyError):
        build_typed_instruction("colour")


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q,expected", [
    ("How many cats are visible?", "counting"),
    ("How many people are in the room?", "counting"),
    ("Why is the umbrella open indoors?", "reasoning"),
    ("How does the system work?", "reasoning"),
    ("What is to the left of the red cup?", "spatial"),
    ("What is above the table?", "spatial"),
    ("What does the sign say?", "text_ocr"),
    ("What text is written on the board?", "text_ocr"),
    ("What is the person doing?", "actions"),
    ("What are they doing with the ball?", "actions"),
    ("What objects are visible?", "objects"),
    ("What items are on the table?", "objects"),
])
def test_classify_common_questions(q: str, expected: str) -> None:
    assert classify_qa_type(q) == expected


def test_classify_ambiguous_falls_back() -> None:
    # no keyword fires → defaults to "objects"
    assert classify_qa_type("Describe the scene.") == "objects"


def test_classify_accuracy_on_30_samples() -> None:
    # 30-sample hand-labeled micro-benchmark → require >= 80% accuracy
    # (5 per type × 6 types = 30)
    labeled: list[tuple[str, str]] = [
        # objects (5)
        ("What objects are on the desk?", "objects"),
        ("What items can you see?", "objects"),
        ("Name the things on the floor.", "objects"),
        ("List the items in the box.", "objects"),
        ("What objects are visible?", "objects"),
        # spatial (5)
        ("What is to the left of the chair?", "spatial"),
        ("What is above the window?", "spatial"),
        ("What is behind the door?", "spatial"),
        ("What is in front of the building?", "spatial"),
        ("What is between the two trees?", "spatial"),
        # actions (5)
        ("What is the man doing?", "actions"),
        ("What are the children doing?", "actions"),
        ("What is she doing at the kitchen?", "actions"),
        ("What are they doing with the equipment?", "actions"),
        ("What is the athlete doing on the field?", "actions"),
        # counting (5)
        ("How many dogs are there?", "counting"),
        ("How many windows does the house have?", "counting"),
        ("How many players are on the field?", "counting"),
        ("How many apples are in the bowl?", "counting"),
        ("How many red cars are visible?", "counting"),
        # text_ocr (5)
        ("What does the sign say?", "text_ocr"),
        ("What is written on the wall?", "text_ocr"),
        ("What text appears on the screen?", "text_ocr"),
        ("What does the label say?", "text_ocr"),
        ("What word is on the box?", "text_ocr"),
        # reasoning (5)
        ("Why is the ground wet?", "reasoning"),
        ("Why are the people running?", "reasoning"),
        ("How does this machine operate?", "reasoning"),
        ("Why is the sky so dark?", "reasoning"),
        ("Why is the door closed?", "reasoning"),
    ]
    correct = sum(1 for q, y in labeled if classify_qa_type(q) == y)
    acc = correct / len(labeled)
    assert acc >= 0.80, f"classifier accuracy {acc:.2f} < 0.80; correct={correct}/30"


# ---------------------------------------------------------------------------
# Distribution + diversity
# ---------------------------------------------------------------------------


def test_distribution_all_zero_on_empty() -> None:
    dist = compute_type_distribution([])
    assert dist == {t: 0.0 for t in QA_TYPES}


def test_distribution_uniform() -> None:
    samples = []
    qs_per_type = {
        "objects": "What objects are visible?",
        "spatial": "What is to the left of it?",
        "actions": "What is he doing?",
        "counting": "How many are there?",
        "text_ocr": "What text is on the sign?",
        "reasoning": "Why is this unusual?",
    }
    for t, q in qs_per_type.items():
        for _ in range(100):
            samples.append(_sample(q))
    dist = compute_type_distribution(samples)
    for t, v in dist.items():
        assert v == pytest.approx(1 / 6, abs=0.02), f"{t}={v}"


def test_diversity_uniform_near_one() -> None:
    dist = {t: 1 / 6 for t in QA_TYPES}
    assert compute_diversity_score(dist) == pytest.approx(1.0, abs=1e-6)


def test_diversity_skewed_low() -> None:
    dist = {t: 0.0 for t in QA_TYPES}
    dist["objects"] = 0.9
    dist["spatial"] = 0.1
    assert compute_diversity_score(dist) < 0.5


def test_diversity_all_zero_is_zero() -> None:
    assert compute_diversity_score({t: 0.0 for t in QA_TYPES}) == 0.0


# ---------------------------------------------------------------------------
# Rebalance
# ---------------------------------------------------------------------------


def test_rebalance_downsamples_excess() -> None:
    samples: list[dict] = []
    # 600 objects + 50 each of other 5 types → 850 total
    for _ in range(600):
        samples.append(_sample("What objects are visible?"))
    for t_q in ["What is to the left?", "What is he doing?", "How many are there?",
                "What does the sign say?", "Why is it this way?"]:
        for _ in range(50):
            samples.append(_sample(t_q))

    out = rebalance_qa_types(samples, max_fraction=0.35, seed=0)
    dist = compute_type_distribution(out)
    assert dist["objects"] <= 0.40  # allow mild slack above max_fraction
    assert len(out) < len(samples)


def test_rebalance_empty_is_empty() -> None:
    assert rebalance_qa_types([]) == []


def test_rebalance_deterministic() -> None:
    samples = [_sample("How many are there?") for _ in range(100)]
    a = rebalance_qa_types(samples, seed=7)
    b = rebalance_qa_types(samples, seed=7)
    assert a == b
