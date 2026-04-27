"""
循环一致性打分器。

实现四路打分：
  S_AR   — 答案重建分 (BERTScore)
  S_CLIP — 跨模态对齐分 (CLIP cosine similarity)
  S_QR   — 问题重建分 (BERTScore)
  S_PPL  — 流畅性分 (Perplexity → sigmoid)

综合分: S_cycle = α·S_AR + β·S_CLIP + γ·S_QR + δ·S_PPL
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image

from tool.model_loader import load_clip, torch
from tool.multimodal_types import MultimodalInferConfig, VQAPair


# ===== BERTScore 批量计算 =====

def bertscore_batch(
    candidates: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "zh",
) -> list[float]:
    """
    计算 BERTScore F1。
    返回 list[float]，每个元素对应一对 (candidate, reference) 的 F1 分数。
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError as exc:
        raise RuntimeError(
            "未安装 bert-score，无法计算循环一致性的 BERTScore。"
            "请在当前 Python 环境中执行: pip install bert-score"
        ) from exc

    if not candidates or not references:
        return []
    P, R, F1 = bert_score_fn(
        candidates, references,
        model_type=model_type, lang=lang,
        verbose=False, batch_size=32,
    )
    return F1.tolist()


# ===== CLIP 图文相似度批量计算 =====

_CLIP_CACHE: dict = {}


def clip_similarity_batch(
    image_paths: list[str],
    texts: list[str],
    clip_model_path: str = "openai/clip-vit-large-patch14-336",
) -> list[float]:
    """
    计算 CLIP cos(img_emb, txt_emb)。
    模型只加载一次并缓存在模块级 _CLIP_CACHE 中。
    """
    if clip_model_path not in _CLIP_CACHE:
        proc, model = load_clip(model_path=clip_model_path)
        _CLIP_CACHE[clip_model_path] = (proc, model)
    proc, model = _CLIP_CACHE[clip_model_path]
    device = next(model.parameters()).device

    scores: list[float] = []
    batch_size = 16

    for start in range(0, len(image_paths), batch_size):
        batch_imgs = image_paths[start:start + batch_size]
        batch_txts = texts[start:start + batch_size]

        pil_images = [Image.open(p).convert("RGB") for p in batch_imgs]
        inputs = proc(
            text=batch_txts, images=pil_images,
            return_tensors="pt", padding=True, truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            img_emb = outputs.image_embeds  # (B, D)
            txt_emb = outputs.text_embeds   # (B, D)

        # 逐对余弦相似度
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        cos_sim = (img_emb * txt_emb).sum(dim=-1)  # (B,)
        scores.extend(cos_sim.cpu().tolist())

    return scores


# ===== PPL → sigmoid 打分 =====

def ppl_to_score(ppls: list[float], mu: float | None = None, tau: float = 5.0) -> list[float]:
    """
    将困惑度列表转为 [0, 1] 分数。
    S_PPL = sigmoid(-(PPL - μ) / τ)
    μ 默认取列表均值。
    """
    if not ppls:
        return []
    arr = np.array(ppls, dtype=np.float64)
    if mu is None:
        mu = float(arr.mean())
    raw = -((arr - mu) / tau)
    return (1.0 / (1.0 + np.exp(-raw))).tolist()


# ===== 综合打分 =====

def compute_cycle_scores(
    vqa_pairs: list[VQAPair],
    reconstructed_answers: list[str],
    reconstructed_questions: list[str] | None,
    ppls: list[float] | None,
    cfg: MultimodalInferConfig,
) -> list[VQAPair]:
    """
    计算四路分数并写入 vqa.cycle_scores，返回更新后的 VQAPair 列表。

    参数:
        vqa_pairs:              原始 VQA 三元组
        reconstructed_answers:  反向验证生成的答案 A' (长度同 vqa_pairs)
        reconstructed_questions: 文本 A2Q 重建的问题 Q' (可选；None 则 S_QR=0)
        ppls:                   MLLM 在 (I,Q) 条件下生成 A 的困惑度 (可选；None 则 S_PPL=0.5)
        cfg:                    包含权重和阈值的配置
    """
    n = len(vqa_pairs)

    # 1. S_AR: BERTScore(A, A')
    original_answers = [v.answer for v in vqa_pairs]
    ar_scores = bertscore_batch(
        reconstructed_answers, original_answers,
        model_type=cfg.bertscore_model,
        lang=cfg.bertscore_lang,
    )

    # 2. S_CLIP: cos(CLIP_img(I), CLIP_txt(A))
    image_paths = [v.image_path for v in vqa_pairs]
    clip_scores = clip_similarity_batch(image_paths, original_answers, cfg.clip_model_path)

    # 3. S_QR: BERTScore(Q, Q')
    if reconstructed_questions is not None:
        original_questions = [v.question for v in vqa_pairs]
        qr_scores = bertscore_batch(
            reconstructed_questions, original_questions,
            model_type=cfg.bertscore_model,
            lang=cfg.bertscore_lang,
        )
    else:
        qr_scores = [0.0] * n

    # 4. S_PPL
    if ppls is not None:
        ppl_scores = ppl_to_score(ppls)
    else:
        ppl_scores = [0.5] * n

    # 5. 加权综合
    for i, vqa in enumerate(vqa_pairs):
        ar = ar_scores[i]
        clip = clip_scores[i]
        qr = qr_scores[i]
        ppl = ppl_scores[i]
        composite = (
            cfg.alpha_ar * ar
            + cfg.beta_clip * clip
            + cfg.gamma_qr * qr
            + cfg.delta_ppl * ppl
        )
        vqa.cycle_scores = {
            "ar": round(ar, 4),
            "clip": round(clip, 4),
            "qr": round(qr, 4),
            "ppl": round(ppl, 4),
            "composite": round(composite, 4),
        }

    return vqa_pairs
