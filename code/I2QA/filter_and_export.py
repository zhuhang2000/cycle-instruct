"""
Stage 3 — 过滤与导出：按循环一致性分数过滤，输出 LlamaFactory 兼容的 ShareGPT JSON。

过滤条件:
  - S_cycle (composite) ≥ cycle_threshold
  - S_AR ≥ ar_threshold
  - S_CLIP ≥ clip_threshold
  - S_QR ≥ qr_threshold

用法:
    python filter_and_export.py -i vqa_scored.json -o training_data.json
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
from tool.multimodal_types import (
    MultimodalInferConfig,
    VQAPair,
    vqa_from_dict,
    vqa_to_sharegpt,
)


def passes_thresholds(vqa: VQAPair, cfg: MultimodalInferConfig) -> bool:
    """检查 VQA 对是否通过所有循环一致性阈值。"""
    s = vqa.cycle_scores
    if not s:
        return False
    return (
        s.get("composite", 0) >= cfg.cycle_threshold
        and s.get("ar", 0) >= cfg.ar_threshold
        and s.get("clip", 0) >= cfg.clip_threshold
        and s.get("qr", 0) >= cfg.qr_threshold
    )


def filter_and_export(
    vqa_pairs: list[VQAPair],
    cfg: MultimodalInferConfig,
    output_path: Path,
) -> list[dict]:
    """
    过滤 + 导出为 LlamaFactory ShareGPT 格式。

    返回:
        保留的 ShareGPT 训练记录列表
    """
    kept = [v for v in vqa_pairs if passes_thresholds(v, cfg)]
    records = [vqa_to_sharegpt(v, include_metadata=True) for v in kept]
    save_json(output_path, records)
    return records


def print_stats(total: int, kept: int, vqa_pairs: list[VQAPair]) -> None:
    """打印过滤统计信息。"""
    import numpy as np
    print(f"\n{'='*50}")
    print(f"过滤统计")
    print(f"{'='*50}")
    print(f"  输入总数:   {total}")
    print(f"  保留数量:   {kept}")
    print(f"  过滤比例:   {(total - kept) / total * 100:.1f}%" if total > 0 else "  N/A")

    if vqa_pairs:
        composites = [v.cycle_scores.get("composite", 0) for v in vqa_pairs if v.cycle_scores]
        if composites:
            arr = np.array(composites)
            print(f"\n  综合分分布:")
            print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}")
            print(f"    min={arr.min():.4f}  max={arr.max():.4f}")
            for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
                pct = (arr >= t).sum() / len(arr) * 100
                print(f"    ≥ {t}: {pct:.1f}%")
    print(f"{'='*50}\n")


# ===== CLI =====

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: 循环一致性过滤 + 导出")
    parser.add_argument("-i", "--input", required=True, help="Stage 2 输出的 scored VQAPair JSON")
    parser.add_argument("-o", "--output", required=True, help="过滤后 LlamaFactory ShareGPT JSON")
    parser.add_argument("--cycle-threshold", type=float, default=0.70)
    parser.add_argument("--ar-threshold", type=float, default=0.60)
    parser.add_argument("--clip-threshold", type=float, default=0.20)
    parser.add_argument("--qr-threshold", type=float, default=0.55)
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    vqa_pairs = [vqa_from_dict(d) for d in raw]

    cfg = MultimodalInferConfig(
        cycle_threshold=args.cycle_threshold,
        ar_threshold=args.ar_threshold,
        clip_threshold=args.clip_threshold,
        qr_threshold=args.qr_threshold,
    )

    output_path = Path(args.output)
    records = filter_and_export(vqa_pairs, cfg, output_path)

    print_stats(total=len(vqa_pairs), kept=len(records), vqa_pairs=vqa_pairs)
    print(f"[Done] 导出 {len(records)} 条训练数据 -> {output_path}")
