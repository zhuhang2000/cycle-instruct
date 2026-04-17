"""
Stage 1 — 正向 VQA 生成：Image → (Q, A) 对。

遵循 generate_pseudo_a.py 的 build_messages / to_record 模式，
调用 multimodal_infer.generate_multimodal() 进行批量推理。

用法:
    python generate_vqa_pairs.py -i samples.json -o vqa_pairs.json
"""

import json
import re
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

from tool.multimodal_infer import generate_multimodal
from tool.multimodal_types import (
    ImageTextSample,
    MultimodalInferConfig,
    VQAPair,
    sample_from_dict,
    vqa_to_dict,
)


# ===== Prompt 构建 =====

SYSTEM_PROMPT = (
    "You are a visual question-answer pair generator specialized in domain-specific images. "
    "Given an image (and optional context), generate ONE high-quality question-answer pair.\n"
    "Requirements:\n"
    "- The question must require understanding the image content to answer\n"
    "- The answer must be grounded in what is visually present\n"
    "- Be specific: include key entities, quantities, relationships, or processes\n"
    "- Avoid generic or trivially answerable questions\n"
    "- Question: at least 10 words; Answer: 1–5 sentences"
)


def build_vqa_messages(sample: ImageTextSample) -> tuple[list[dict], list[str]]:
    """构建多模态 chat messages + 图片路径列表。"""
    user_content = "<image>\n"
    if sample.source_text:
        user_content += f"Context: {sample.source_text}\n\n"
    user_content += (
        "Based on the image above, generate ONE question-answer pair.\n"
        "Use this exact format:\n"
        "Q: [your question]\n"
        "A: [your answer]"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages, [sample.image_path]


# ===== 输出解析 =====

_QA_PATTERN = re.compile(
    r"Q:\s*(?P<question>.+?)\s*\nA:\s*(?P<answer>.+)",
    re.DOTALL,
)


def parse_qa(raw: str) -> tuple[str, str]:
    """从模型输出中解析 Q/A 对。"""
    m = _QA_PATTERN.search(raw)
    if m:
        return m.group("question").strip(), m.group("answer").strip()
    # 回退：按换行分割
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if len(lines) >= 2:
        return lines[0], " ".join(lines[1:])
    return raw.strip(), ""


# 模块级变量，在 CLI __main__ 中由 cfg.mllm_model_path 设置
_generation_model: str = ""


def to_record(sample: ImageTextSample, raw_output: str) -> dict:
    """将 (sample, model_output) 转为 VQAPair dict。"""
    q, a = parse_qa(raw_output)
    vqa = VQAPair(
        image_path=sample.image_path,
        image_id=sample.image_id,
        question=q,
        answer=a,
        generation_model=_generation_model,
    )
    return vqa_to_dict(vqa)


# ===== CLI =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Image → VQA pairs")
    parser.add_argument("-i", "--input", required=True, help="输入 ImageTextSample JSON 路径")
    parser.add_argument("-o", "--output", required=True, help="输出 VQAPair JSON 路径")
    parser.add_argument("-bk", "--backend", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("-q", "--quantization", default=None)
    parser.add_argument("-m", "--model-path", default="", help="MLLM 模型路径")
    parser.add_argument("-n", "--num-qa", type=int, default=3, help="每张图生成 QA 对数")
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    samples = [sample_from_dict(d) for d in raw_data]

    # 每张图重复 N 次以生成多组 QA（不同 temperature 下）
    expanded: list[ImageTextSample] = []
    for s in samples:
        expanded.extend([s] * args.num_qa)

    cfg = MultimodalInferConfig(
        backend=args.backend,
        quantization=args.quantization,
        mllm_model_path=args.model_path,
        num_qa_per_image=args.num_qa,
        temperature=0.7,  # 多样性生成
        top_p=0.9,
    )

    _generation_model = args.model_path

    generate_multimodal(expanded, build_vqa_messages, Path(args.output), to_record, cfg=cfg)
