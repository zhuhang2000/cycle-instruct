"""Hallucination metrics: CHAIRi / CHAIRs + CLIP-low-alignment rate.

The detector is pluggable (``Detector = (image_path, candidate_names) ->
dict[name, bool]``). In production this is OWL-ViT v2 via transformers;
in tests it's a lambda over a fixture dict.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from experiments.intrinsic._io import (
    extract_answer,
    extract_image_path,
    load_vqa_jsonl,
    save_json,
)
from experiments.intrinsic.base import IntrinsicMetric, register_metric


Detector = Callable[[str, list[str]], dict[str, bool]]


# Naive fallback noun extraction when spaCy is unavailable. Keeps alpha
# tokens of length ≥ 4, drops an English stop-word list.
_STOPWORDS = {
    "this", "that", "these", "those", "there", "their", "they", "them",
    "what", "which", "where", "when", "with", "from", "into", "onto",
    "have", "been", "being", "does", "doing", "done", "said", "says",
    "some", "many", "much", "more", "most", "less", "least", "none",
    "about", "above", "below", "under", "over", "around", "between",
    "image", "photo", "picture", "scene", "visible", "shown", "showing",
    "yes", "no", "not",
}


def extract_noun_phrases(text: str, use_spacy: bool = True) -> list[str]:
    """Return lowercased candidate noun phrases from ``text``.

    Falls back to alpha-token heuristic if spaCy isn't installed.
    """
    if use_spacy:
        try:
            import spacy  # noqa: WPS433
            try:
                nlp = extract_noun_phrases._nlp  # type: ignore[attr-defined]
            except AttributeError:
                nlp = spacy.blank("en")
                # Blank model has no parser so noun_chunks won't work; try
                # loading the small English model, else fall through.
                try:
                    nlp = spacy.load("en_core_web_sm")
                except Exception:  # noqa: BLE001
                    nlp = None
                extract_noun_phrases._nlp = nlp  # type: ignore[attr-defined]
            if nlp is not None:
                doc = nlp(text)
                chunks = {chunk.root.lemma_.lower() for chunk in doc.noun_chunks}
                chunks = {c for c in chunks if c.isalpha() and len(c) >= 3
                          and c not in _STOPWORDS}
                if chunks:
                    return sorted(chunks)
        except ImportError:
            pass
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return sorted({t for t in tokens if t not in _STOPWORDS})


def _make_owlvit_detector(model_name: str = "google/owlv2-base-patch16-ensemble",
                          device: str = "cpu",
                          threshold: float = 0.2) -> Detector:
    """Build an OWL-ViT v2 zero-shot detector function."""
    from transformers import (  # noqa: WPS433
        Owlv2Processor,
        Owlv2ForObjectDetection,
    )
    from PIL import Image  # noqa: WPS433
    import torch  # noqa: WPS433

    processor = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()

    def _detect(image_path: str, candidates: list[str]) -> dict[str, bool]:
        if not candidates:
            return {}
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:  # noqa: BLE001
            return {c: False for c in candidates}
        inputs = processor(
            text=[candidates], images=image, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold,
        )[0]
        present = set()
        for label_idx in results["labels"].tolist():
            present.add(candidates[label_idx])
        return {c: (c in present) for c in candidates}

    return _detect


def compute_chair(
    samples: Iterable[dict],
    detector: Detector,
    *,
    use_spacy: bool = True,
) -> dict[str, Any]:
    """CHAIRi (instance) and CHAIRs (sentence) hallucination rates.

    * CHAIRi = hallucinated mentions / total mentions
    * CHAIRs = answers with ≥1 hallucinated mention / total answers
    """
    total_mentions = 0
    halluc_mentions = 0
    total_answers = 0
    halluc_answers = 0
    per_sample: list[dict] = []
    per_category: dict[str, dict[str, int]] = {}

    for s in samples:
        answer = extract_answer(s)
        image_path = extract_image_path(s)
        if not answer:
            continue
        total_answers += 1
        nouns = extract_noun_phrases(answer, use_spacy=use_spacy)
        if not nouns:
            per_sample.append({"image_path": image_path, "nouns": [],
                               "hallucinated": [], "chair_i": 0.0})
            continue
        if image_path:
            presence = detector(image_path, nouns)
        else:
            presence = {n: False for n in nouns}
        hallucinated = [n for n in nouns if not presence.get(n, False)]
        total_mentions += len(nouns)
        halluc_mentions += len(hallucinated)
        if hallucinated:
            halluc_answers += 1
        for n in nouns:
            pc = per_category.setdefault(n, {"total": 0, "halluc": 0})
            pc["total"] += 1
            if n in hallucinated:
                pc["halluc"] += 1
        per_sample.append({
            "image_path": image_path,
            "nouns": nouns,
            "hallucinated": hallucinated,
            "chair_i": len(hallucinated) / len(nouns) if nouns else 0.0,
        })

    return {
        "chairi": (halluc_mentions / total_mentions) if total_mentions else 0.0,
        "chairs": (halluc_answers / total_answers) if total_answers else 0.0,
        "total_mentions": total_mentions,
        "hallucinated_mentions": halluc_mentions,
        "total_answers": total_answers,
        "hallucinated_answers": halluc_answers,
        "per_sample": per_sample,
        "per_category": per_category,
    }


def clip_low_align_rate(
    samples: list[dict],
    threshold: float = 0.20,
    clip_model_path: str = "openai/clip-vit-large-patch14-336",
) -> dict[str, Any]:
    """Fraction of samples whose CLIP(I, answer) < ``threshold``."""
    try:
        from tool.cycle_scorer import clip_similarity_batch  # noqa: WPS433
    except ImportError as exc:
        return {"error": f"clip_similarity_batch unavailable: {exc}",
                "num_samples": len(samples)}
    imgs: list[str] = []
    answers: list[str] = []
    for s in samples:
        ip = extract_image_path(s)
        ans = extract_answer(s)
        if ip and ans:
            imgs.append(ip)
            answers.append(ans)
    if not imgs:
        return {"rate": 0.0, "num_samples": 0, "num_low": 0}
    try:
        scores = clip_similarity_batch(imgs, answers, clip_model_path=clip_model_path)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"clip scoring failed: {exc}", "num_samples": len(imgs)}
    low = [s for s in scores if s < threshold]
    return {
        "rate": len(low) / len(scores),
        "num_samples": len(scores),
        "num_low": len(low),
        "mean_score": sum(scores) / len(scores),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Metric wrapper
# ---------------------------------------------------------------------------

@register_metric("hallucination")
class HallucinationMetric(IntrinsicMetric):
    """Module-3 metric: CHAIRi/CHAIRs + CLIP low-align rate."""

    requires_images = True
    requires_gpu = True

    def compute(
        self,
        samples: list[dict],
        *,
        detector: Detector | None = None,
        detector_name: str = "owlv2",
        device: str = "cpu",
        threshold: float = 0.2,
        run_clip_crosscheck: bool = False,
        clip_threshold: float = 0.20,
        use_spacy: bool = True,
        **_: Any,
    ) -> dict[str, Any]:
        if detector is None:
            if detector_name == "none":
                def detector(img, names):  # type: ignore[assignment]
                    return {n: False for n in names}
            else:
                try:
                    detector = _make_owlvit_detector(
                        device=device, threshold=threshold,
                    )
                except Exception as exc:  # noqa: BLE001
                    return {"error": f"detector unavailable: {exc}"}
        chair = compute_chair(samples, detector, use_spacy=use_spacy)
        out: dict[str, Any] = {
            "chairi": chair["chairi"],
            "chairs": chair["chairs"],
            "total_mentions": chair["total_mentions"],
            "hallucinated_mentions": chair["hallucinated_mentions"],
            "total_answers": chair["total_answers"],
            "hallucinated_answers": chair["hallucinated_answers"],
            "per_category": chair["per_category"],
        }
        if run_clip_crosscheck:
            out["clip_low_align"] = clip_low_align_rate(samples, clip_threshold)
        return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hallucination metrics.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--detector", type=str, default="owlv2")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--clip-crosscheck", action="store_true")
    args = p.parse_args(argv)

    samples = load_vqa_jsonl(args.input)
    if args.image_dir is not None:
        for s in samples:
            ip = extract_image_path(s)
            if ip and not Path(ip).is_absolute():
                s["image_path"] = str(args.image_dir / ip)
    result = HallucinationMetric().compute(
        samples,
        detector_name=args.detector,
        device=args.device,
        threshold=args.threshold,
        run_clip_crosscheck=args.clip_crosscheck,
    )
    if args.out:
        save_json(args.out, result)
    else:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
