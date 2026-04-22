"""Diversity metrics: distinct-n, Self-BLEU, TTR, MTLD, embedding spread.

All pairwise metrics use a random ``n_sample`` pair subset (default 200)
to keep cost O(n_sample) on large pools.
"""
from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from experiments.intrinsic._io import (
    extract_answer,
    extract_image_path,
    extract_question,
    load_vqa_jsonl,
    save_json,
)
from experiments.intrinsic.base import IntrinsicMetric, register_metric


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Text statistics
# ---------------------------------------------------------------------------

def distinct_n(texts: Iterable[str], n: int = 2) -> float:
    """Fraction of unique n-grams over all n-grams.

    Returns 0.0 when no text has at least ``n`` tokens. For a single
    repeated text with T tokens and L = T-n+1 n-grams, this equals 1/L.
    """
    ngrams: list[tuple[str, ...]] = []
    for t in texts:
        toks = _tokenize(t)
        if len(toks) < n:
            continue
        ngrams.extend(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def type_token_ratio(texts: Iterable[str]) -> float:
    toks: list[str] = []
    for t in texts:
        toks.extend(_tokenize(t))
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def _mtld_one_direction(tokens: list[str], threshold: float = 0.72) -> float:
    """MTLD inner loop — number of factors over the stream."""
    if not tokens:
        return 0.0
    factor_count = 0.0
    types: set[str] = set()
    token_count = 0
    for tok in tokens:
        types.add(tok)
        token_count += 1
        ttr = len(types) / token_count
        if ttr <= threshold:
            factor_count += 1
            types = set()
            token_count = 0
    if token_count > 0:
        ttr = len(types) / token_count
        remain = (1.0 - ttr) / (1.0 - threshold) if threshold < 1 else 0.0
        factor_count += remain
    if factor_count == 0:
        return float(len(tokens))
    return len(tokens) / factor_count


def mtld(text: str, threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010)."""
    tokens = _tokenize(text)
    if len(tokens) < 2:
        return 0.0
    forward = _mtld_one_direction(tokens, threshold)
    backward = _mtld_one_direction(list(reversed(tokens)), threshold)
    return (forward + backward) / 2


def _sentence_bleu4(hyp_tokens: list[str], ref_tokens: list[str]) -> float:
    """Smoothed BLEU-4 for a single pair (no brevity penalty weighting trick)."""
    if not hyp_tokens or not ref_tokens:
        return 0.0
    weights = [0.25] * 4
    log_precisions = 0.0
    for n in range(1, 5):
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        if not hyp_ngrams:
            return 0.0
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        total = sum(hyp_ngrams.values())
        # Add-one smoothing for zero n-gram matches (method 1 of Chen & Cherry).
        p = (overlap + 1e-9) / (total + 1e-9) if overlap > 0 else 1e-9
        log_precisions += weights[n - 1] * math.log(p)
    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(
        1 - len(ref_tokens) / max(len(hyp_tokens), 1)
    )
    return bp * math.exp(log_precisions)


def self_bleu(texts: list[str], n_sample: int = 200, seed: int = 0) -> float:
    """Average BLEU-4 of a sampled text against one other sampled text.

    For perfectly identical inputs this returns 1.0; for fully disjoint
    token sets it approaches 0.0. We sample ``n_sample`` (hyp, ref) pairs
    to avoid O(n²) cost on large pools.
    """
    texts = [t for t in texts if t and t.strip()]
    if len(texts) < 2:
        return 0.0
    rng = random.Random(seed)
    tokenized = [_tokenize(t) for t in texts]
    n = len(tokenized)
    total = 0.0
    count = 0
    pairs = min(n_sample, n * (n - 1))
    for _ in range(pairs):
        i = rng.randrange(n)
        j = rng.randrange(n)
        while j == i:
            j = rng.randrange(n)
        total += _sentence_bleu4(tokenized[i], tokenized[j])
        count += 1
    if count == 0:
        return 0.0
    return total / count


def length_std(texts: Iterable[str]) -> float:
    lens = [len(_tokenize(t)) for t in texts]
    if len(lens) < 2:
        return 0.0
    mean = sum(lens) / len(lens)
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Embedding-based + image diversity (best-effort — skipped if deps missing)
# ---------------------------------------------------------------------------

def _pairwise_cosine_mean(embs, n_sample: int = 200, seed: int = 0) -> float:
    try:
        import numpy as np
    except ImportError:
        return float("nan")
    arr = np.asarray(embs, dtype="float32")
    if arr.ndim != 2 or arr.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    arr = arr / norms
    rng = random.Random(seed)
    n = arr.shape[0]
    total = 0.0
    count = 0
    for _ in range(min(n_sample, n * (n - 1))):
        i = rng.randrange(n)
        j = rng.randrange(n)
        while j == i:
            j = rng.randrange(n)
        total += 1.0 - float(arr[i] @ arr[j])
        count += 1
    return total / count if count else 0.0


def _phash_for(path: str) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(path).convert("L").resize((8, 8))
    except Exception:  # noqa: BLE001
        return None
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = "".join("1" if p >= avg else "0" for p in pixels)
    return bits


def phash_diversity(image_paths: list[str]) -> float:
    """Fraction of unique 8x8 mean-hashes; 1.0 = every image distinct."""
    hashes: list[str] = []
    for p in image_paths:
        h = _phash_for(p)
        if h is not None:
            hashes.append(h)
    if not hashes:
        return 0.0
    return len(set(hashes)) / len(hashes)


# ---------------------------------------------------------------------------
# Metric wrapper
# ---------------------------------------------------------------------------

@register_metric("diversity")
class DiversityMetric(IntrinsicMetric):
    """Module-2 metric: lexical + optional embedding / image diversity."""

    requires_images = False  # only True if image-side metrics are requested

    def compute(
        self,
        samples: list[dict],
        *,
        n_sample_self_bleu: int = 200,
        seed: int = 0,
        with_images: bool = False,
        embedding_model: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        questions = [extract_question(s) for s in samples]
        answers = [extract_answer(s) for s in samples]
        combined = [f"{q} {a}".strip() for q, a in zip(questions, answers)]

        result: dict[str, Any] = {
            "num_samples": len(samples),
            "distinct_1_q": distinct_n(questions, 1),
            "distinct_2_q": distinct_n(questions, 2),
            "distinct_3_q": distinct_n(questions, 3),
            "distinct_1_a": distinct_n(answers, 1),
            "distinct_2_a": distinct_n(answers, 2),
            "distinct_3_a": distinct_n(answers, 3),
            "type_token_ratio_q": type_token_ratio(questions),
            "type_token_ratio_a": type_token_ratio(answers),
            "question_length_std": length_std(questions),
            "answer_length_std": length_std(answers),
            "self_bleu_4_q": self_bleu(questions, n_sample_self_bleu, seed),
            "self_bleu_4_a": self_bleu(answers, n_sample_self_bleu, seed),
        }
        mtld_vals = [mtld(t) for t in combined if t]
        result["mtld_mean"] = (sum(mtld_vals) / len(mtld_vals)) if mtld_vals else 0.0

        if embedding_model:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: WPS433
                model = SentenceTransformer(embedding_model)
                embs = model.encode(combined, show_progress_bar=False)
                result["embedding_diversity"] = _pairwise_cosine_mean(
                    embs, n_sample=n_sample_self_bleu, seed=seed,
                )
            except Exception as exc:  # noqa: BLE001
                result["embedding_diversity_error"] = str(exc)

        if with_images:
            image_paths = [extract_image_path(s) for s in samples]
            image_paths = [p for p in image_paths if p]
            result["image_phash_unique_ratio"] = phash_diversity(image_paths)
            result["num_images"] = len(image_paths)

        return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Diversity metrics for VQA JSONL.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--n-sample", type=int, default=200)
    p.add_argument("--with-images", action="store_true")
    p.add_argument("--embedding-model", type=str, default=None)
    args = p.parse_args(argv)

    samples = load_vqa_jsonl(args.input)
    result = DiversityMetric().compute(
        samples,
        n_sample_self_bleu=args.n_sample,
        with_images=args.with_images,
        embedding_model=args.embedding_model,
    )
    if args.out:
        save_json(args.out, result)
    else:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
