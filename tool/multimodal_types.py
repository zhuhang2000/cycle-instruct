"""
多模态 Cycle-Instruct 共享数据类型。

定义 ImageTextSample / VQAPair / MultimodalInferConfig，
供正向生成、反向验证、循环打分、过滤导出等模块共用。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from tool.chat_infer import InferConfig


# ===== 数据样本 =====

@dataclass
class ImageTextSample:
    """从原始垂域数据（PDF / Web / 文档）清洗后得到的图文对。"""
    image_path: str                         # 图片绝对/相对路径
    image_id: str = ""                      # 唯一标识（默认由 image_path 哈希生成）
    source_text: str | None = None          # OCR 文本 / caption / 上下文段落
    source_type: str = "unknown"            # "pdf_scan" | "web_image" | "document_figure"
    metadata: dict = field(default_factory=dict)  # 页码、URL、bounding box 等

    def __post_init__(self) -> None:
        if not self.image_id:
            self.image_id = hashlib.md5(self.image_path.encode()).hexdigest()[:12]


@dataclass
class VQAPair:
    """由 MLLM 生成的 VQA 三元组 (Image, Question, Answer)，附带循环一致性分数。"""
    image_path: str
    image_id: str
    question: str
    answer: str
    generation_model: str = ""
    cycle_scores: dict[str, float] = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        return self.cycle_scores.get("composite", 0.0)


# ===== 多模态推理配置 =====

@dataclass
class MultimodalInferConfig(InferConfig):
    """继承 InferConfig，扩展多模态、循环一致性相关参数。"""

    # --- 图像处理 ---
    image_max_pixels: int = 768 * 768
    image_min_pixels: int = 32 * 32

    # --- MLLM 配置 ---
    mllm_model_path: str = ""               # 多模态 LLM 路径（Qwen-VL / LLaVA / InternVL）
    verifier_model_path: str = ""           # 验证器模型路径（空则复用 mllm_model_path）

    # --- CLIP 配置 ---
    clip_model_path: str = "openai/clip-vit-large-patch14-336"

    # --- BERTScore 配置 ---
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    bertscore_lang: str = "zh"               # BERTScore 语言（"zh", "en" 等）

    # --- 生成参数 ---
    num_qa_per_image: int = 3               # 每张图生成多少组 QA

    # --- 循环一致性打分权重 ---
    alpha_ar: float = 0.40                  # 答案重建分 (Answer Reconstruction)
    beta_clip: float = 0.25                 # CLIP 跨模态对齐分
    gamma_qr: float = 0.25                  # 问题重建分 (Question Reconstruction)
    delta_ppl: float = 0.10                 # PPL 流畅性分

    # --- 过滤阈值 ---
    cycle_threshold: float = 0.70           # 综合分下限
    ar_threshold: float = 0.60
    clip_threshold: float = 0.20
    qr_threshold: float = 0.55

    def effective_verifier_path(self) -> str:
        return self.verifier_model_path or self.mllm_model_path


# ===== 序列化工具 =====

def sample_to_dict(s: ImageTextSample) -> dict:
    return {
        "image_path": s.image_path,
        "image_id": s.image_id,
        "source_text": s.source_text,
        "source_type": s.source_type,
        "metadata": s.metadata,
    }


def sample_from_dict(d: dict) -> ImageTextSample:
    return ImageTextSample(
        image_path=d["image_path"],
        image_id=d.get("image_id", ""),
        source_text=d.get("source_text"),
        source_type=d.get("source_type", "unknown"),
        metadata=d.get("metadata", {}),
    )


def vqa_to_dict(v: VQAPair) -> dict:
    return {
        "image_path": v.image_path,
        "image_id": v.image_id,
        "question": v.question,
        "answer": v.answer,
        "generation_model": v.generation_model,
        "cycle_scores": v.cycle_scores,
    }


def vqa_from_dict(d: dict) -> VQAPair:
    return VQAPair(
        image_path=d["image_path"],
        image_id=d.get("image_id", ""),
        question=d["question"],
        answer=d["answer"],
        generation_model=d.get("generation_model", ""),
        cycle_scores=d.get("cycle_scores", {}),
    )


def vqa_to_sharegpt(v: VQAPair, *, include_metadata: bool = False) -> dict:
    """转换为 LlamaFactory ShareGPT 多模态格式（对齐 mllm_demo.json）。"""
    record = {
        "messages": [
            {"role": "user", "content": f"<image>{v.question}"},
            {"role": "assistant", "content": v.answer},
        ],
        "images": [v.image_path],
    }
    if include_metadata:
        record.update(
            {
                "image_id": v.image_id,
                "question": v.question,
                "answer": v.answer,
                "generation_model": v.generation_model,
                "cycle_scores": v.cycle_scores,
                "cycle_score": v.composite_score,
            }
        )
    return record
