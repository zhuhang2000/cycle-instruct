"""Aggregator: dispatch enabled intrinsic metrics, write report + plots.

Loads a YAML config (see ``experiments/configs/intrinsic_default.yaml``)
describing which modules to run and their kwargs, then produces:

* ``intrinsic_report.json`` — flat JSON of every metric's output
* ``intrinsic_report.md``   — human-readable summary tables
* ``plots/*.png``           — per-module charts, if matplotlib is available
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from experiments.intrinsic import METRIC_REGISTRY  # noqa: F401 (trigger registration)
from experiments.intrinsic._io import (
    extract_image_path,
    load_vqa_jsonl,
    save_json,
    save_md_table,
)
from experiments.intrinsic.base import METRIC_REGISTRY as _REG

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    import yaml  # type: ignore
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_image_paths(samples: list[dict], image_dir: Path | None) -> None:
    if image_dir is None:
        return
    for s in samples:
        ip = extract_image_path(s)
        if ip and not Path(ip).is_absolute() and not Path(ip).exists():
            s["image_path"] = str(image_dir / ip)


def _render_markdown(report: dict) -> str:
    parts: list[str] = [f"# Intrinsic Quality Report\n",
                        f"**num_samples**: {report.get('num_samples', 'N/A')}\n"]

    qa = report.get("qa_types")
    if qa and "type_distribution" in qa:
        parts.append("\n## QA type distribution\n")
        rows = [(k, f"{v:.3f}") for k, v in qa["type_distribution"].items()]
        parts.append(save_md_table(rows, ["type", "fraction"]))
        parts.append(f"\n- diversity_score: {qa.get('diversity_score', 'N/A'):.3f}")
        if "js_divergence_vs_seed" in qa:
            parts.append(f"\n- js_divergence_vs_seed: "
                         f"{qa['js_divergence_vs_seed']:.3f}")

    div = report.get("diversity")
    if div:
        parts.append("\n\n## Diversity\n")
        rows = [
            ["distinct_2_q", f"{div.get('distinct_2_q', 0):.3f}"],
            ["distinct_2_a", f"{div.get('distinct_2_a', 0):.3f}"],
            ["self_bleu_4_q", f"{div.get('self_bleu_4_q', 0):.3f}"],
            ["self_bleu_4_a", f"{div.get('self_bleu_4_a', 0):.3f}"],
            ["mtld_mean", f"{div.get('mtld_mean', 0):.2f}"],
        ]
        parts.append(save_md_table(rows, ["metric", "value"]))

    hal = report.get("hallucination")
    if hal and "chairi" in hal:
        parts.append("\n\n## Hallucination\n")
        rows = [
            ["CHAIRi", f"{hal.get('chairi', 0):.3f}"],
            ["CHAIRs", f"{hal.get('chairs', 0):.3f}"],
            ["total_mentions", hal.get("total_mentions", 0)],
            ["total_answers", hal.get("total_answers", 0)],
        ]
        parts.append(save_md_table(rows, ["metric", "value"]))

    cyc = report.get("cycle_stats")
    if cyc and "components" in cyc:
        parts.append("\n\n## Cycle scores (composite percentiles)\n")
        comp = cyc["components"].get("composite") or {}
        rows = [
            ["mean", f"{comp.get('mean', 0):.3f}"],
            ["p25", f"{comp.get('p25', 0):.3f}"],
            ["p50", f"{comp.get('p50', 0):.3f}"],
            ["p75", f"{comp.get('p75', 0):.3f}"],
            ["p95", f"{comp.get('p95', 0):.3f}"],
        ]
        parts.append(save_md_table(rows, ["stat", "value"]))

    ling = report.get("linguistic")
    if ling:
        parts.append("\n\n## Linguistic quality\n")
        rows = [
            ["answer_length_mean",
             f"{ling.get('answer_length', {}).get('mean', 0):.2f}"],
            ["answer_length_std",
             f"{ling.get('answer_length', {}).get('std', 0):.2f}"],
            ["template_repeat_rate_q",
             f"{ling.get('template_repeat_rate_q', 0):.3f}"],
            ["yes_no_answer_rate",
             f"{ling.get('yes_no_answer_rate', 0):.3f}"],
        ]
        parts.append(save_md_table(rows, ["metric", "value"]))

    alignment = report.get("alignment")
    if alignment and "clip_image_answer_mean" in alignment:
        parts.append("\n\n## Image-text alignment\n")
        rows = [
            ["clip_image_answer_mean",
             f"{alignment['clip_image_answer_mean']:.3f}"],
            ["blind_caption_rate",
             f"{alignment.get('blind_caption_rate', 0):.3f}"],
            ["mi_shuffle_estimate",
             f"{alignment.get('mi_shuffle_estimate', 0):.3f}"],
        ]
        parts.append(save_md_table(rows, ["metric", "value"]))

    return "\n".join(parts) + "\n"


def run_report(
    samples: list[dict],
    *,
    modules: dict[str, dict[str, Any]],
    out_dir: Path,
    seed_ref: list[dict] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"num_samples": len(samples)}
    for name, cfg in modules.items():
        if not cfg.get("enabled", True):
            continue
        cls = _REG.get(name)
        if cls is None:
            logger.warning("metric %r not registered; skipping", name)
            report[name] = {"status": "not_registered"}
            continue
        metric = cls()
        kwargs = {k: v for k, v in cfg.items() if k != "enabled"}
        if name == "qa_types" and seed_ref is not None:
            kwargs.setdefault("seed_ref", seed_ref)
        try:
            result = metric.compute(samples, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("metric %s failed: %s", name, exc)
            report[name] = {"error": str(exc)}
            continue
        report[name] = result
        try:
            metric.plots(result, out_dir / "plots")
        except Exception as exc:  # noqa: BLE001
            logger.warning("plots for %s failed: %s", name, exc)

    save_json(out_dir / "intrinsic_report.json", report)
    md_path = out_dir / "intrinsic_report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_modules_arg(raw: str, all_names: list[str]) -> dict[str, dict[str, Any]]:
    if not raw or raw == "all":
        return {name: {"enabled": True} for name in all_names}
    return {name.strip(): {"enabled": True} for name in raw.split(",")
            if name.strip()}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Run intrinsic metric suite.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--seed-ref", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("report"))
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config; overrides --modules when set.")
    p.add_argument("--modules", type=str, default="all")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--smoke", action="store_true",
                   help="skip hallucination + alignment (no GPU), cap to 100 samples")
    args = p.parse_args(argv)

    samples = load_vqa_jsonl(args.input)
    if args.smoke:
        samples = samples[:100]
    _resolve_image_paths(samples, args.image_dir)

    seed_ref = load_vqa_jsonl(args.seed_ref) if args.seed_ref else None

    if args.config is not None:
        cfg = _load_yaml(args.config)
        modules = cfg.get("modules", {})
    else:
        modules = _parse_modules_arg(args.modules, list(_REG.keys()))
    if args.smoke:
        modules.setdefault("hallucination", {"enabled": False})["enabled"] = False
        modules.setdefault("alignment", {"enabled": False})["enabled"] = False

    for mod_cfg in modules.values():
        mod_cfg.setdefault("device", args.device)

    run_report(samples, modules=modules, out_dir=args.out, seed_ref=seed_ref)
    logger.info("[intrinsic] wrote report to %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
