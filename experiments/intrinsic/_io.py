"""Shared I/O helpers for intrinsic metric modules."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def load_vqa_jsonl(path: Path | str) -> list[dict]:
    """Load a JSONL of VQA records (ShareGPT or legacy VQAPair dicts)."""
    out: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract_question(sample: dict) -> str:
    """Pull the user question out of either ShareGPT or VQAPair schemas."""
    if "question" in sample and sample["question"]:
        return str(sample["question"])
    for msg in sample.get("messages", []):
        if msg.get("role") == "user":
            txt = str(msg.get("content", ""))
            return re.sub(r"<image>\s*", "", txt).strip()
    return ""


def extract_answer(sample: dict) -> str:
    """Pull the assistant answer out of either ShareGPT or VQAPair schemas."""
    if "answer" in sample and sample["answer"]:
        return str(sample["answer"])
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def extract_image_path(sample: dict) -> str:
    """Pull the primary image path out of either schema."""
    if sample.get("image_path"):
        return str(sample["image_path"])
    imgs = sample.get("images") or []
    if imgs:
        return str(imgs[0])
    return ""


def save_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def save_md_table(rows: Iterable[list[str]], headers: list[str]) -> str:
    """Render a list of rows as a GitHub-flavored markdown table."""
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


__all__ = [
    "load_vqa_jsonl",
    "extract_question",
    "extract_answer",
    "extract_image_path",
    "save_json",
    "save_md_table",
]
