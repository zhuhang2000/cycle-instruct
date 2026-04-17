"""
去重 + 质量门控。

三层过滤:
  1. 图像去重: pHash + Hamming 距离
  2. 文本去重: MinHash + LSH (Jaccard)
  3. 质量过滤: 模糊度、信息熵、文本长度

用法:
    python dedup_and_filter.py -i samples.jsonl -o samples_clean.jsonl
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _ensure_project_root_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "tool").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_ensure_project_root_on_path()

from tool.multimodal_types import ImageTextSample, sample_from_dict, sample_to_dict


# ===== 图像去重: pHash =====

def _phash(img: Image.Image, hash_size: int = 8) -> int:
    """计算感知哈希 (pHash)。返回 64-bit 整数。"""
    # 缩放到 (hash_size+1) x hash_size 的灰度图
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    # 水平差分
    diff = pixels[:, 1:] > pixels[:, :-1]
    # 转为整数哈希
    return int(diff.flatten().dot(1 << np.arange(diff.size, dtype=np.uint64)))


def _hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def dedup_images(
    samples: list[ImageTextSample],
    phash_threshold: int = 8,
) -> list[ImageTextSample]:
    """基于 pHash 图像去重，Hamming 距离 < threshold 的视为重复。"""
    seen_hashes: list[int] = []
    unique: list[ImageTextSample] = []

    for s in samples:
        try:
            img = Image.open(s.image_path)
            h = _phash(img)
        except Exception:
            continue

        is_dup = False
        for seen in seen_hashes:
            if _hamming_distance(h, seen) < phash_threshold:
                is_dup = True
                break

        if not is_dup:
            seen_hashes.append(h)
            unique.append(s)

    return unique


# ===== 文本去重: MinHash =====

def _shingles(text: str, k: int = 3) -> set[str]:
    """提取 k-shingle 集合。"""
    text = text.strip()
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def _minhash_signature(shingle_set: set[str], num_hashes: int = 128) -> list[int]:
    """计算 MinHash 签名。"""
    sig = [float("inf")] * num_hashes
    for s in shingle_set:
        for i in range(num_hashes):
            h = int(hashlib.md5(f"{i}_{s}".encode()).hexdigest(), 16) & 0xFFFFFFFF
            sig[i] = min(sig[i], h)
    return sig


def _jaccard_from_minhash(sig1: list[int], sig2: list[int]) -> float:
    """从 MinHash 签名估算 Jaccard 相似度。"""
    return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)


def dedup_texts(
    samples: list[ImageTextSample],
    jaccard_threshold: float = 0.8,
) -> list[ImageTextSample]:
    """基于 MinHash 文本去重。"""
    if not samples:
        return []

    sigs = []
    for s in samples:
        text = s.source_text or ""
        shingles = _shingles(text)
        sigs.append(_minhash_signature(shingles))

    unique: list[ImageTextSample] = []
    unique_sigs: list[list[int]] = []

    for i, (s, sig) in enumerate(zip(samples, sigs)):
        is_dup = False
        for usig in unique_sigs:
            if _jaccard_from_minhash(sig, usig) >= jaccard_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(s)
            unique_sigs.append(sig)

    return unique


# ===== 质量过滤 =====

def _is_blurry(img: Image.Image, threshold: float = 100.0) -> bool:
    """Laplacian 方差检测模糊度。"""
    gray = np.array(img.convert("L"), dtype=np.float64)
    # Laplacian kernel
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from scipy.signal import convolve2d
    lap = convolve2d(gray, laplacian, mode="same", boundary="symm")
    return float(lap.var()) < threshold


def _image_entropy(img: Image.Image) -> float:
    """计算图像信息熵。"""
    gray = np.array(img.convert("L"))
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def filter_quality(
    samples: list[ImageTextSample],
    blur_threshold: float = 100.0,
    entropy_threshold: float = 3.0,
    min_text_length: int = 20,
) -> list[ImageTextSample]:
    """质量过滤：模糊度、信息熵、文本长度。"""
    passed: list[ImageTextSample] = []

    for s in samples:
        # 文本长度
        if s.source_text and len(s.source_text.strip()) < min_text_length:
            continue

        # 图像质量
        try:
            img = Image.open(s.image_path)
        except Exception:
            continue

        if _is_blurry(img, threshold=blur_threshold):
            continue
        if _image_entropy(img) < entropy_threshold:
            continue

        passed.append(s)

    return passed


# ===== 组合入口 =====

def deduplicate_and_filter(
    samples: list[ImageTextSample],
    phash_threshold: int = 8,
    jaccard_threshold: float = 0.8,
    blur_threshold: float = 100.0,
    entropy_threshold: float = 3.0,
    min_text_length: int = 20,
) -> list[ImageTextSample]:
    """
    三层过滤：图像去重 → 文本去重 → 质量门控。
    """
    n0 = len(samples)

    # 1. 图像去重
    samples = dedup_images(samples, phash_threshold)
    n1 = len(samples)

    # 2. 文本去重
    samples = dedup_texts(samples, jaccard_threshold)
    n2 = len(samples)

    # 3. 质量过滤
    samples = filter_quality(samples, blur_threshold, entropy_threshold, min_text_length)
    n3 = len(samples)

    print(f"[Dedup+Filter] {n0} → 图像去重 {n1} → 文本去重 {n2} → 质量过滤 {n3}")
    return samples


# ===== CLI =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="去重 + 质量门控")
    parser.add_argument("-i", "--input", required=True, help="输入 JSONL (ImageTextSample)")
    parser.add_argument("-o", "--output", required=True, help="输出 JSONL")
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--jaccard-threshold", type=float, default=0.8)
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--entropy-threshold", type=float, default=3.0)
    parser.add_argument("--min-text-length", type=int, default=20)
    args = parser.parse_args()

    # 读取
    samples: list[ImageTextSample] = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(sample_from_dict(json.loads(line)))

    # 过滤
    cleaned = deduplicate_and_filter(
        samples,
        phash_threshold=args.phash_threshold,
        jaccard_threshold=args.jaccard_threshold,
        blur_threshold=args.blur_threshold,
        entropy_threshold=args.entropy_threshold,
        min_text_length=args.min_text_length,
    )

    # 保存
    with Path(args.output).open("w", encoding="utf-8") as f:
        for s in cleaned:
            f.write(json.dumps(sample_to_dict(s), ensure_ascii=False) + "\n")

    print(f"[Done] 清洗后保留 {len(cleaned)} 条 -> {args.output}")
