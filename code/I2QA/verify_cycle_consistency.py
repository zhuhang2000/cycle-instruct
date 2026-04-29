"""
Stage 2 — 循环一致性验证：对 Stage 1 生成的 VQA 对进行三路验证打分。

验证路径:
  2a. 答案重建: Image + Q → A'，BERTScore(A, A')
  2b. CLIP 对齐: cos(CLIP_img(I), CLIP_txt(A))
  2c. 问题重建: Image + A → Q'，BERTScore(Q, Q')

用法:
    python verify_cycle_consistency.py -i vqa_pairs.json -o vqa_scored.json
"""

import json
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "tool").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_ensure_project_root_on_path()

from tool.chat_infer import save_json
from tool.cycle_scorer import compute_cycle_scores
from tool.multimodal_infer import generate_multimodal
from tool.multimodal_types import (
    ImageTextSample,
    MultimodalInferConfig,
    VQAPair,
    vqa_from_dict,
    vqa_to_dict,
)


# ===== 2a. 答案重建：Image + Q → A' =====

def _build_answer_verify_messages(vqa: VQAPair) -> tuple[list[dict], list[str]]:
    """给 verifier MLLM 的 prompt：看图回答问题。"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise visual question answering assistant. "
                "Answer the question based ONLY on what you can see in the image. "
                "Be concise and factual."
            ),
        },
        {
            "role": "user",
            "content": f"<image>\n{vqa.question}",
        },
    ]
    return messages, [vqa.image_path]


def _vqa_to_sample(vqa: VQAPair) -> ImageTextSample:
    """VQAPair → ImageTextSample，用于 generate_multimodal 的输入。"""
    return ImageTextSample(
        image_path=vqa.image_path,
        image_id=vqa.image_id,
        source_text=vqa.question,  # 把 question 放入 source_text 以传递
    )


def reconstruct_answers(
    vqa_pairs: list[VQAPair], cfg: MultimodalInferConfig, tmp_dir: Path,
) -> list[str]:
    """
    2a. 用 verifier MLLM 重新回答 (Image, Q)，返回重建答案列表。
    """
    def build_verify_messages(sample: ImageTextSample) -> tuple[list[dict], list[str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise visual question answering assistant. "
                    "Answer the question based ONLY on what you can see in the image. "
                    "Be concise and factual."
                ),
            },
            {
                "role": "user",
                "content": f"<image>\n{sample.source_text}",
            },
        ]
        return messages, [sample.image_path]

    def to_record_raw(sample: ImageTextSample, output: str) -> dict:
        return {"image_id": sample.image_id, "question": sample.source_text, "answer_prime": output}

    # 使用 verifier 模型；未显式配置时回退到主模型路径。
    verifier_model_path = cfg.effective_verifier_path() or cfg.model_path
    verify_cfg = MultimodalInferConfig(
        backend=cfg.backend,
        quantization=cfg.quantization,
        model_path=verifier_model_path,
        mllm_model_path=verifier_model_path,
        temperature=0.0,  # 确定性生成
        max_new_tokens=cfg.max_new_tokens,
        batch_size=cfg.batch_size,
        save_every=cfg.save_every,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )

    samples = [_vqa_to_sample(v) for v in vqa_pairs]
    output_path = tmp_dir / "answer_reconstruction.json"
    records = generate_multimodal(samples, build_verify_messages, output_path, to_record_raw, cfg=verify_cfg)
    return [r.get("answer_prime", "") for r in records]


# ===== 2c. 问题重建：Image + A → Q' =====

def _vqa_to_answer_sample(vqa: VQAPair) -> ImageTextSample:
    """VQAPair → ImageTextSample，用 answer 作为条件文本重建问题。"""
    return ImageTextSample(
        image_path=vqa.image_path,
        image_id=vqa.image_id,
        source_text=vqa.answer,
        source_type="question_reconstruction",
    )


def reconstruct_questions(
    vqa_pairs: list[VQAPair], cfg: MultimodalInferConfig, tmp_dir: Path,
) -> list[str]:
    """
    2c. 用 verifier MLLM 从 (Image, Answer) 重建 Question。
    """
    def build_question_messages(sample: ImageTextSample) -> tuple[list[dict], list[str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise visual question generation assistant. "
                    "Given an image and a target answer, write ONE natural question "
                    "that is answerable from the visible image content and whose answer "
                    "would be the target answer. Do not use outside knowledge."
                ),
            },
            {
                "role": "user",
                "content": (
                    "<image>\n"
                    f"Target answer: {sample.source_text}\n\n"
                    "Generate only the question. The question must depend on the image."
                ),
            },
        ]
        return messages, [sample.image_path]

    def to_record_q(sample: ImageTextSample, question: str) -> dict:
        return {
            "image_id": sample.image_id,
            "answer": sample.source_text,
            "question_prime": question,
        }

    verifier_model_path = cfg.effective_verifier_path() or cfg.model_path
    question_cfg = MultimodalInferConfig(
        backend=cfg.backend,
        quantization=cfg.quantization,
        model_path=verifier_model_path,
        mllm_model_path=verifier_model_path,
        temperature=0.0,
        max_new_tokens=cfg.max_new_tokens,
        batch_size=cfg.batch_size,
        save_every=cfg.save_every,
        tensor_parallel_size=cfg.tensor_parallel_size,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        disable_log=cfg.disable_log,
        dtype_gpu=cfg.dtype_gpu,
        dtype_cpu=cfg.dtype_cpu,
    )

    samples = [_vqa_to_answer_sample(v) for v in vqa_pairs]
    output_path = tmp_dir / "question_reconstruction.json"
    records = generate_multimodal(
        samples,
        build_question_messages,
        output_path,
        to_record_q,
        cfg=question_cfg,
    )
    return [r.get("question_prime", "") for r in records]


# ===== 主验证流程 =====

def verify_batch(
    vqa_pairs: list[VQAPair], cfg: MultimodalInferConfig, tmp_dir: Path,
) -> list[VQAPair]:
    """
    对 VQA 对执行三路循环验证，计算综合分数。
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 2a. 答案重建
    reconstructed_answers = reconstruct_answers(vqa_pairs, cfg, tmp_dir)

    # 2c. 问题重建：Image + A -> Q'
    reconstructed_questions = reconstruct_questions(vqa_pairs, cfg, tmp_dir)

    # 综合打分（2b CLIP 在 cycle_scorer 内部计算）
    scored = compute_cycle_scores(
        vqa_pairs,
        reconstructed_answers=reconstructed_answers,
        reconstructed_questions=reconstructed_questions,
        ppls=None,  # PPL 在此简化版中暂不计算，默认 0.5
        cfg=cfg,
    )
    return scored


# ===== CLI =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: 循环一致性验证")
    parser.add_argument("-i", "--input", required=True, help="Stage 1 输出的 VQAPair JSON")
    parser.add_argument("-o", "--output", required=True, help="带分数的 VQAPair JSON 输出")
    parser.add_argument("-bk", "--backend", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("-q", "--quantization", default=None)
    parser.add_argument("-m", "--model-path", default="", help="MLLM 模型路径（生成器）")
    parser.add_argument("-vm", "--verifier-model-path", default="", help="Verifier MLLM 路径（默认同 -m）")
    parser.add_argument("--text-model-path", default="", help="兼容旧参数；Image+A 问题重建已改用 verifier MLLM")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    vqa_pairs = [vqa_from_dict(d) for d in raw]

    cfg = MultimodalInferConfig(
        backend=args.backend,
        quantization=args.quantization,
        mllm_model_path=args.model_path,
        verifier_model_path=args.verifier_model_path,
        model_path=args.text_model_path or args.model_path,
    )

    output_path = Path(args.output)
    tmp_dir = output_path.parent / "tmp_verify"

    scored = verify_batch(vqa_pairs, cfg, tmp_dir)

    save_json(output_path, [vqa_to_dict(v) for v in scored])
    print(f"[Done] 验证完成，{len(scored)} 条已打分 -> {output_path}")
