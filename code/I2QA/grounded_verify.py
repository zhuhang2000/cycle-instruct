"""
Stage 2.5 — Grounded Verifier: verify whether VQA answers are supported by images.

This script is intentionally standalone. It reads scored VQA records, asks a VLM to
judge each (image, question, answer), and writes the original records enriched with
grounded verification fields and a final keep flag.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def _ensure_project_root_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "tool").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_ensure_project_root_on_path()

from tool.chat_infer import save_json
from tool.multimodal_infer import generate_multimodal
from tool.multimodal_types import ImageTextSample, MultimodalInferConfig


PROMPT_VERSION = "strict_v2"

SYSTEM_PROMPT = """You are a strict visual grounding verifier.

Your job is NOT to decide whether the answer sounds plausible.
Your job is to decide whether every claim in the answer is directly supported by visible evidence in the image.

Rules:
1. If any part of the answer is wrong, unsupported, or unverifiable from the image, set supported=false.
2. Do not use outside knowledge unless the image explicitly contains the needed label, number, arrow, table entry, or text.
3. For counting questions, independently count the visible objects. If the count differs, supported=false.
4. For charts, tables, diagrams, maps, and Punnett squares, verify the exact labels, entries, arrows, positions, and numeric values.
5. For probability or math questions, recompute the result. If the answer's number is inconsistent with the image, supported=false.
6. Questions asking about purpose, function, habitat, cause, or intended use should be unsupported unless the image explicitly states that information.
7. If your explanation contradicts the answer, supported=false.
8. If uncertain, supported=false.

Return only valid JSON:
{
  "supported": true or false,
  "error_type": "none/counting_error/math_error/table_error/diagram_error/unsupported_inference/contradiction/uncertain/other",
  "reason": "brief explanation",
  "corrected_answer": "corrected answer if possible, otherwise null"
}"""


_ALLOWED_ERROR_TYPES = {
    "none",
    "counting_error",
    "math_error",
    "table_error",
    "diagram_error",
    "unsupported_inference",
    "contradiction",
    "uncertain",
    "other",
}

_NUMBER_RE = re.compile(r"\b\d+/\d+\b|\d+(?:\.\d+)?%?")
_CONTRADICTION_RE = re.compile(
    r"\b(contradict|contradicts|inconsistent|differs|different|wrong|incorrect|"
    r"not correct|not supported|should be|rather than|instead of)\b",
    re.IGNORECASE,
)

_RISK_PATTERNS: dict[str, tuple[str, ...]] = {
    "counting": (
        "how many",
        "number of",
        "count",
        "counts",
        "total number",
    ),
    "math": (
        "probability",
        "percent",
        "percentage",
        "%",
        "ratio",
        "total",
        "difference",
        "compare",
        "kinetic energy",
    ),
    "diagram": (
        "punnett",
        "food web",
        "chart",
        "table",
        "diagram",
        "lattice",
        "allele",
        "genotype",
        "phenotype",
    ),
    "inference": (
        "purpose",
        "function",
        "used for",
        "use for",
        "habitat",
        "environment",
        "type of",
        "kind of",
        "cause",
        "why",
        "might",
        "likely",
        "indicating",
    ),
}


def _sample_from_record(record: dict[str, Any]) -> ImageTextSample:
    payload = {
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
    }
    return ImageTextSample(
        image_path=record["image_path"],
        image_id=record.get("image_id", ""),
        source_text=json.dumps(payload, ensure_ascii=False),
        source_type="grounded_verify",
        metadata={"record": record},
    )


def build_grounded_verify_messages(sample: ImageTextSample) -> tuple[list[dict], list[str]]:
    payload = json.loads(sample.source_text or "{}")
    user_content = (
        "<image>\n"
        f"Question: {payload.get('question', '')}\n"
        f"Answer: {payload.get('answer', '')}\n"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, [sample.image_path]


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))

    if not isinstance(obj, dict):
        raise ValueError("verifier output is not a JSON object")
    return obj


def _normalize_verdict(raw: str) -> dict[str, Any]:
    try:
        obj = _extract_json_object(raw)
    except Exception as exc:
        return {
            "supported": False,
            "error_type": "other",
            "reason": f"Could not parse verifier JSON: {exc}",
            "corrected_answer": None,
            "raw_output": raw,
        }

    supported = obj.get("supported")
    if isinstance(supported, str):
        supported = supported.strip().lower() == "true"
    else:
        supported = bool(supported)

    error_type = str(obj.get("error_type", "none" if supported else "other")).strip()
    if error_type not in _ALLOWED_ERROR_TYPES:
        error_type = "other"
    if not supported and error_type == "none":
        error_type = "other"

    corrected_answer = obj.get("corrected_answer")

    return {
        "supported": supported,
        "error_type": error_type,
        "reason": str(obj.get("reason", "")).strip(),
        "corrected_answer": corrected_answer,
        "raw_output": raw,
    }


def _numbers(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(_NUMBER_RE.findall(str(text).lower()))


def _detect_risk_types(question: str, answer: str) -> list[str]:
    text = f"{question}\n{answer}".lower()
    risks = [
        risk_type
        for risk_type, patterns in _RISK_PATTERNS.items()
        if any(pattern in text for pattern in patterns)
    ]
    return risks


def _apply_post_checks(record: dict[str, Any]) -> dict[str, Any]:
    verdict = dict(record.get("grounded_verification") or {})
    question = str(record.get("question", ""))
    answer = str(record.get("answer", ""))
    reason = str(verdict.get("reason", ""))
    corrected_answer = verdict.get("corrected_answer")
    error_type = str(verdict.get("error_type", "other"))
    verifier_version = str(verdict.get("verifier_version", "legacy"))

    supported = verdict.get("supported") is True
    manual_review = False
    review_reasons: list[str] = []
    post_supported = supported

    risk_types = _detect_risk_types(question, answer)

    if corrected_answer not in (None, ""):
        post_supported = False
        review_reasons.append("verifier_provided_corrected_answer")

    if error_type != "none":
        post_supported = False
        review_reasons.append("verifier_error_type_not_none")

    answer_nums = _numbers(answer)
    reason_nums = _numbers(reason)
    corrected_nums = _numbers(str(corrected_answer) if corrected_answer is not None else "")

    if reason_nums and answer_nums and not (answer_nums & reason_nums):
        manual_review = True
        review_reasons.append("answer_reason_numeric_mismatch")

    if corrected_nums and answer_nums and not (answer_nums & corrected_nums):
        post_supported = False
        manual_review = True
        review_reasons.append("answer_corrected_numeric_mismatch")

    if supported and _CONTRADICTION_RE.search(reason):
        manual_review = True
        review_reasons.append("supported_reason_contains_contradiction_language")

    if risk_types and verifier_version != PROMPT_VERSION:
        manual_review = True
        review_reasons.append("high_risk_not_verified_by_strict_v2")

    requires_independent_count_check = "counting" in risk_types
    if requires_independent_count_check:
        manual_review = True
        review_reasons.append("requires_independent_count_check")

    high_risk_unverified = bool(risk_types and (not post_supported or manual_review))

    verdict["verifier_version"] = verifier_version
    verdict["post_supported"] = bool(post_supported)
    verdict["manual_review"] = bool(manual_review)
    verdict["risk_types"] = risk_types
    verdict["requires_independent_count_check"] = bool(requires_independent_count_check)
    verdict["high_risk_unverified"] = bool(high_risk_unverified)
    verdict["post_check"] = {
        "answer_numbers": sorted(answer_nums),
        "reason_numbers": sorted(reason_nums),
        "corrected_answer_numbers": sorted(corrected_nums),
        "review_reasons": review_reasons,
    }
    record["grounded_verification"] = verdict
    return record


def _passes_score_filters(
    record: dict[str, Any],
    *,
    min_composite: float,
    min_ar: float,
    min_qr: float,
) -> bool:
    scores = record.get("cycle_scores") or {}
    return (
        float(scores.get("composite", 0.0)) >= min_composite
        and float(scores.get("ar", 0.0)) >= min_ar
        and float(scores.get("qr", 0.0)) >= min_qr
    )


def to_grounded_record(sample: ImageTextSample, raw_output: str) -> dict[str, Any]:
    record = dict(sample.metadata["record"])
    verdict = _normalize_verdict(raw_output)
    verdict["verifier_version"] = PROMPT_VERSION
    record["grounded_verification"] = verdict
    record = _apply_post_checks(record)
    return record


def apply_keep_flags(
    records: list[dict[str, Any]],
    *,
    min_composite: float = 0.55,
    min_ar: float = 0.55,
    min_qr: float = 0.55,
) -> list[dict[str, Any]]:
    for record in records:
        grounded = record.get("grounded_verification") or {}
        score_ok = _passes_score_filters(
            record,
            min_composite=min_composite,
            min_ar=min_ar,
            min_qr=min_qr,
        )
        risk_types = grounded.get("risk_types") or []
        grounded_ok = (
            grounded.get("post_supported") is True
            and grounded.get("error_type") == "none"
            and grounded.get("corrected_answer") is None
            and grounded.get("manual_review") is False
            and "inference" not in risk_types
            and grounded.get("requires_independent_count_check") is not True
        )
        keep_final = bool(score_ok and grounded_ok)
        record["keep_score_grounded"] = bool(
            score_ok and grounded.get("supported") is True
        )
        record["keep_final"] = keep_final
        record["keep"] = keep_final
    return records


def grounded_verify(
    records: list[dict[str, Any]],
    output_path: Path,
    cfg: MultimodalInferConfig,
    *,
    min_composite: float = 0.55,
    min_ar: float = 0.55,
    min_qr: float = 0.55,
) -> list[dict[str, Any]]:
    samples = [_sample_from_record(record) for record in records]
    verified = generate_multimodal(
        samples,
        build_grounded_verify_messages,
        output_path,
        to_grounded_record,
        cfg=cfg,
    )
    verified = apply_keep_flags(
        verified,
        min_composite=min_composite,
        min_ar=min_ar,
        min_qr=min_qr,
    )
    save_json(output_path, verified)
    return verified


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2.5: grounded visual QA verification")
    parser.add_argument("-i", "--input", required=True, help="Stage 2 scored VQA JSON")
    parser.add_argument("-o", "--output", required=True, help="Grounded verification output JSON")
    parser.add_argument("-bk", "--backend", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("-q", "--quantization", default=None)
    parser.add_argument("-m", "--model-path", required=True, help="Verifier MLLM model path")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-composite", type=float, default=0.55)
    parser.add_argument("--min-ar", type=float, default=0.55)
    parser.add_argument("--min-qr", type=float, default=0.55)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Do not run the VLM; recompute post-checks and keep flags for an existing grounded output.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if args.postprocess_only:
        verified = [
            _apply_post_checks(dict(record))
            if record.get("grounded_verification")
            else dict(record)
            for record in records
        ]
        verified = apply_keep_flags(
            verified,
            min_composite=args.min_composite,
            min_ar=args.min_ar,
            min_qr=args.min_qr,
        )
        save_json(output_path, verified)
    else:
        cfg = MultimodalInferConfig(
            backend=args.backend,
            quantization=args.quantization,
            mllm_model_path=args.model_path,
            temperature=0.0,
            max_new_tokens=args.max_new_tokens,
            save_every=args.save_every,
            batch_size=args.batch_size,
        )
        verified = grounded_verify(
            records,
            output_path,
            cfg,
            min_composite=args.min_composite,
            min_ar=args.min_ar,
            min_qr=args.min_qr,
        )

    kept = sum(1 for record in verified if record.get("keep_final"))
    supported = sum(
        1
        for record in verified
        if (record.get("grounded_verification") or {}).get("supported") is True
    )
    post_supported = sum(
        1
        for record in verified
        if (record.get("grounded_verification") or {}).get("post_supported") is True
    )
    manual_review = sum(
        1
        for record in verified
        if (record.get("grounded_verification") or {}).get("manual_review") is True
    )
    print(
        f"[Done] grounded verification complete: "
        f"{supported}/{len(verified)} verifier-supported, "
        f"{post_supported}/{len(verified)} post-supported, "
        f"{manual_review}/{len(verified)} manual-review, "
        f"{kept}/{len(verified)} kept -> {output_path}"
    )


if __name__ == "__main__":
    main()
