"""
Web 图文 → ImageTextSample 提取。

流程:
  1. 解析 HTML，提取 <img> + alt / figcaption / 相邻文本
  2. 下载图片，校验分辨率和长宽比
  3. CLIP 预对齐过滤（可选）

用法:
    python web_extractor.py -i pages.jsonl -o ./output_dir/
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

from PIL import Image


def _ensure_project_root_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "tool").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_ensure_project_root_on_path()

from tool.multimodal_types import ImageTextSample, sample_to_dict


def _parse_html_images(html: str, base_url: str = "") -> list[dict]:
    """
    从 HTML 中提取图片 URL 及关联文本。

    返回 list[dict]，每项包含:
        - src: 图片 URL
        - text: 关联文本（alt, figcaption, 或相邻段落）
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results = []

    for img_tag in soup.find_all("img"):
        src = img_tag.get("src", "")
        if not src:
            continue
        if base_url:
            src = urljoin(base_url, src)

        # 收集关联文本（优先级：figcaption > alt > 相邻段落）
        text = ""

        # figcaption
        figure = img_tag.find_parent("figure")
        if figure:
            caption = figure.find("figcaption")
            if caption:
                text = caption.get_text(strip=True)

        # alt
        if not text:
            text = img_tag.get("alt", "").strip()

        # 相邻段落
        if not text:
            sibling = img_tag.find_next_sibling("p")
            if sibling:
                text = sibling.get_text(strip=True)[:300]

        if text:
            results.append({"src": src, "text": text})

    return results


def _download_image(url: str, save_dir: Path, timeout: int = 15) -> str | None:
    """下载图片到本地。返回保存路径，失败返回 None。"""
    import requests

    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
            return None

        img_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ext = ".jpg"
        for candidate in [".png", ".webp", ".gif", ".jpeg"]:
            if candidate in url.lower():
                ext = candidate
                break

        save_path = save_dir / f"{img_hash}{ext}"
        save_path.write_bytes(resp.content)
        return str(save_path)
    except Exception:
        return None


def _validate_image(
    img_path: str,
    min_size: int = 224,
    max_aspect_ratio: float = 5.0,
) -> bool:
    """校验图片尺寸和长宽比。"""
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            if w < min_size or h < min_size:
                return False
            ratio = max(w, h) / max(min(w, h), 1)
            if ratio > max_aspect_ratio:
                return False
            return True
    except Exception:
        return False


def extract_from_web(
    html_path_or_content: str,
    output_dir: str,
    base_url: str = "",
    clip_filter: bool = False,
    clip_threshold: float = 0.15,
) -> list[ImageTextSample]:
    """
    从 Web HTML 提取 ImageTextSample。

    参数:
        html_path_or_content: HTML 文件路径或 HTML 字符串
        output_dir:           图片输��目录
        base_url:             用于解析相对 URL
        clip_filter:          是否启用 CLIP 预对齐过滤
        clip_threshold:       CLIP 相似度阈值

    返回:
        ImageTextSample 列表
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # 读取 HTML
    p = Path(html_path_or_content)
    if p.exists():
        html = p.read_text(encoding="utf-8")
        if not base_url:
            base_url = ""
    else:
        html = html_path_or_content

    # 提取图文对
    raw_pairs = _parse_html_images(html, base_url)

    # 下载并校验
    samples: list[ImageTextSample] = []
    for pair in raw_pairs:
        local_path = _download_image(pair["src"], img_dir)
        if local_path is None:
            continue
        if not _validate_image(local_path):
            Path(local_path).unlink(missing_ok=True)
            continue

        sample = ImageTextSample(
            image_path=local_path,
            source_text=pair["text"],
            source_type="web_image",
            metadata={"url": pair["src"]},
        )
        samples.append(sample)

    # 可选 CLIP 过滤
    if clip_filter and samples:
        from tool.cycle_scorer import clip_similarity_batch
        paths = [s.image_path for s in samples]
        texts = [s.source_text or "" for s in samples]
        scores = clip_similarity_batch(paths, texts)
        samples = [s for s, sc in zip(samples, scores) if sc >= clip_threshold]

    return samples


# ===== CLI =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web HTML → ImageTextSample 提取")
    parser.add_argument("-i", "--input", required=True, help="HTML 文件路径或 JSONL（每行 {url, html}）")
    parser.add_argument("-o", "--output-dir", required=True, help="输���目录")
    parser.add_argument("--clip-filter", action="store_true", help="启用 CLIP 预对齐过滤")
    parser.add_argument("--clip-threshold", type=float, default=0.15)
    args = parser.parse_args()

    input_path = Path(args.input)
    all_samples: list[ImageTextSample] = []

    if input_path.suffix == ".jsonl":
        # JSONL 格式：每行 {"url": "...", "html": "..."}
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                samples = extract_from_web(
                    entry.get("html", ""), args.output_dir,
                    base_url=entry.get("url", ""),
                    clip_filter=args.clip_filter,
                    clip_threshold=args.clip_threshold,
                )
                all_samples.extend(samples)
    else:
        all_samples = extract_from_web(
            str(input_path), args.output_dir,
            clip_filter=args.clip_filter,
            clip_threshold=args.clip_threshold,
        )

    jsonl_path = Path(args.output_dir) / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(sample_to_dict(s), ensure_ascii=False) + "\n")

    print(f"[Done] 提取 {len(all_samples)} 个图文对 -> {jsonl_path}")
