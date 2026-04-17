"""
Stage 2 — 循环一致性验证：对 Stage 1 生成的 VQA 对进行三路验证打分。

验证路径:
  2a. 答案重建: Image + Q → A'，BERTScore(A, A')
  2b. CLIP 对齐: cos(CLIP_img(I), CLIP_txt(A))
  2c. 问题重建: A → Q'（复用现有 text A2Q），BERTScore(Q, Q')

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

from tool.chat_infer import generate as text_generate, InferConfig
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

    # 使用 verifier 模型
    verify_cfg = MultimodalInferConfig(
        backend=cfg.backend,
        quantization=cfg.quantization,
        mllm_model_path=cfg.effective_verifier_path(),
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


# ===== 2c. 问题重建：A → Q'（复用现有 text A2Q）=====

def reconstruct_questions(
    vqa_pairs: list[VQAPair], cfg: MultimodalInferConfig, tmp_dir: Path,
) -> list[str]:
    """
    2c. 用现有 text A2Q pipeline 从 answer 重建 question。
    复用 code/Q2A/generate_pseudo_q.py 的 build_messages。
    """
    import importlib.util
    _q2a_path = Path(__file__).resolve().parents[1] / "Q2A" / "generate_pseudo_q.py"
    _spec = importlib.util.spec_from_file_location("generate_pseudo_q", _q2a_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    text_build_messages = _mod.build_messages

    answers = [v.answer for v in vqa_pairs]
    output_path = tmp_dir / "question_reconstruction.json"

    text_cfg = InferConfig(
        model_path=cfg.model_path,
        backend=cfg.backend,
        quantization=cfg.quantization,
        temperature=0.0,
        max_new_tokens=cfg.max_new_tokens,
        batch_size=cfg.batch_size,
        save_every=cfg.save_every,
    )

    def to_record_q(answer: str, question: str) -> dict:
        return {"answer": answer, "question_prime": question}

    records = text_generate(answers, text_build_messages, output_path, to_record_q, cfg=text_cfg)
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

    # 2c. 问题重建（复用现有 text A2Q）
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
    parser.add_argument("--text-model-path", default="", help="文本 LLM 路径（用于 A2Q 问题重建）")
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

    from tool.chat_infer import save_json
    save_json(output_path, [vqa_to_dict(v) for v in scored])
    print(f"[Done] 验证完成，{len(scored)} 条已打分 -> {output_path}")
