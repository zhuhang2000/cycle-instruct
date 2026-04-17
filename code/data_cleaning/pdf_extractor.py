"""
PDF 扫描件 → ImageTextSample 提取流水线。

流程:
  1. pymupdf 渲染每页为高分辨率图像
  2. 版面分析（DocLayout-YOLO / LayoutLMv3）分割图表与文本区域
  3. OCR 提取文本块；对图表区域关联最近 caption
  4. 输出裁切图像 + ImageTextSample JSONL

用���:
    python pdf_extractor.py -i document.pdf -o ./output_dir/
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

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


def _render_pages(pdf_path: str, dpi: int = 300) -> list[tuple[int, Image.Image]]:
    """渲染 PDF 每页为 PIL Image。"""
    import fitz  # pymupdf

    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append((page_num, img))
    doc.close()
    return pages


def _extract_figures_simple(page_img: Image.Image, page_num: int, output_dir: Path) -> list[dict]:
    """
    简化版图表提取：将整页作为一张图保存。

    TODO: 接入 DocLayout-YOLO 或 LayoutLMv3 做精细版面分割，
          分别裁切 figure / table / equation 区域。
    """
    img_hash = hashlib.md5(page_img.tobytes()[:4096]).hexdigest()[:10]
    filename = f"page_{page_num:04d}_{img_hash}.png"
    save_path = output_dir / filename
    page_img.save(save_path)
    return [{"image_path": str(save_path), "page_num": page_num, "region": "full_page"}]


def _ocr_page(page_img: Image.Image) -> str:
    """
    对页面执行 OCR。优先使用 PaddleOCR，回退到 pytesseract。
    """
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        import numpy as np
        result = ocr.ocr(np.array(page_img), cls=True)
        if result and result[0]:
            texts = [line[1][0] for line in result[0] if line[1][0].strip()]
            return "\n".join(texts)
        return ""
    except ImportError:
        pass

    try:
        import pytesseract
        return pytesseract.image_to_string(page_img, lang="chi_sim+eng")
    except ImportError:
        return ""


def _associate_caption(ocr_text: str, page_num: int) -> str | None:
    """
    从 OCR 文本中寻找 caption（包含 "图"/"Figure"/"Table"/"表" 的行）。
    如未找到，返回前 200 字作为上下文。
    """
    import re
    lines = ocr_text.split("\n")
    caption_patterns = [
        re.compile(r"^(图|Figure|Fig\.?|Table|表)\s*\d", re.IGNORECASE),
    ]
    for line in lines:
        for pat in caption_patterns:
            if pat.search(line.strip()):
                return line.strip()
    # 回退：取前 200 字
    text = ocr_text.strip()
    return text[:200] if text else None


def extract_from_pdf(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
) -> list[ImageTextSample]:
    """
    从 PDF 文档提取 ImageTextSample 列表。

    参数:
        pdf_path:    PDF 文件路径
        output_dir:  图片输出目录
        dpi:         渲染分辨率

    返回:
        ImageTextSample 列表
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    pages = _render_pages(pdf_path, dpi=dpi)
    samples: list[ImageTextSample] = []

    for page_num, page_img in pages:
        # 提取图像区域
        figures = _extract_figures_simple(page_img, page_num, img_dir)

        # OCR 提取文本
        ocr_text = _ocr_page(page_img)
        caption = _associate_caption(ocr_text, page_num)

        for fig in figures:
            sample = ImageTextSample(
                image_path=fig["image_path"],
                source_text=caption,
                source_type="pdf_scan",
                metadata={
                    "pdf_path": pdf_path,
                    "page_num": fig["page_num"],
                    "region": fig["region"],
                    "ocr_text_length": len(ocr_text),
                },
            )
            samples.append(sample)

    return samples


# ===== CLI =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF → ImageTextSample 提取")
    parser.add_argument("-i", "--input", required=True, help="PDF 文件路径")
    parser.add_argument("-o", "--output-dir", required=True, help="输出目录")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    samples = extract_from_pdf(args.input, args.output_dir, dpi=args.dpi)

    # 保存 JSONL
    jsonl_path = Path(args.output_dir) / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(sample_to_dict(s), ensure_ascii=False) + "\n")

    print(f"[Done] 提取 {len(samples)} 个图文对 -> {jsonl_path}")
