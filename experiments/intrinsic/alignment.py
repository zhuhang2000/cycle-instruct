"""Image-text alignment metrics (CLIP-based, optional heavy module)."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from experiments.intrinsic._io import (
    extract_answer,
    extract_image_path,
    extract_question,
    load_vqa_jsonl,
    save_json,
)
from experiments.intrinsic.base import IntrinsicMetric, register_metric


def _clip_batch(images: list[str], texts: list[str],
                clip_model_path: str) -> list[float] | None:
    try:
        from tool.cycle_scorer import clip_similarity_batch  # noqa: WPS433
    except ImportError:
        return None
    try:
        return clip_similarity_batch(images, texts, clip_model_path=clip_model_path)
    except Exception:  # noqa: BLE001
        return None


@register_metric("alignment")
class AlignmentMetric(IntrinsicMetric):
    """Module-6 metric: CLIP(image, answer/question) + shuffle-MI estimate."""

    requires_images = True
    requires_gpu = True

    def compute(
        self,
        samples: list[dict],
        *,
        clip_model_path: str = "openai/clip-vit-large-patch14-336",
        blind_threshold: float = 0.15,
        seed: int = 0,
        **_: Any,
    ) -> dict[str, Any]:
        pairs: list[tuple[str, str, str]] = []
        for s in samples:
            img = extract_image_path(s)
            q = extract_question(s)
            a = extract_answer(s)
            if img and a:
                pairs.append((img, q, a))
        if not pairs:
            return {"num_samples": 0, "error": "no samples with image+answer"}

        imgs = [p[0] for p in pairs]
        questions = [p[1] for p in pairs]
        answers = [p[2] for p in pairs]

        ia = _clip_batch(imgs, answers, clip_model_path)
        iq = _clip_batch(imgs, questions, clip_model_path)
        if ia is None:
            return {"error": "clip scorer unavailable", "num_samples": len(pairs)}

        rng = random.Random(seed)
        shuffled = answers[:]
        rng.shuffle(shuffled)
        ia_shuf = _clip_batch(imgs, shuffled, clip_model_path)

        def _mean(xs: list[float] | None) -> float:
            return (sum(xs) / len(xs)) if xs else 0.0

        blind = sum(1 for s in ia if s < blind_threshold) / len(ia)
        return {
            "num_samples": len(pairs),
            "clip_image_answer_mean": _mean(ia),
            "clip_image_question_mean": _mean(iq),
            "blind_caption_rate": blind,
            "mi_shuffle_estimate": _mean(ia) - _mean(ia_shuf),
            "blind_threshold": blind_threshold,
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Image-text alignment metrics.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--clip-model", type=str,
                   default="openai/clip-vit-large-patch14-336")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    samples = load_vqa_jsonl(args.input)
    if args.image_dir is not None:
        for s in samples:
            ip = extract_image_path(s)
            if ip and not Path(ip).is_absolute():
                s["image_path"] = str(args.image_dir / ip)
    result = AlignmentMetric().compute(samples, clip_model_path=args.clip_model)
    if args.out:
        save_json(args.out, result)
    else:
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
