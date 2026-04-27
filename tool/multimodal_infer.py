"""
多模态统一推理引擎。

镜像 chat_infer.py 的 generate() 模式，扩展支持 Image + Text 输入。
支持 vLLM 批量推理和 HuggingFace 逐条推理两种后端。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

from PIL import Image

from tool.chat_infer import (
    _build_prompt,
    _post_process_text,
    save_json,
)
from tool.logging_utils import setup_logger
from tool.model_loader import (
    first_device_of,
    load_mllm,
    torch,
)
from tool.multimodal_types import ImageTextSample, MultimodalInferConfig


# ===== 多模态 HF 推理 =====

def _to_hf_multimodal_messages(messages: list[dict], image_count: int) -> list[dict]:
    """Convert literal <image> placeholders to HF chat-template image items."""
    if image_count <= 0:
        return messages

    converted: list[dict] = []
    placeholder_count = 0
    inserted_fallback = False

    for message in messages:
        content = message.get("content", "")
        new_message = dict(message)

        if isinstance(content, str) and "<image>" in content:
            parts = content.split("<image>")
            items: list[dict] = []
            for idx, part in enumerate(parts):
                if idx > 0:
                    items.append({"type": "image"})
                    placeholder_count += 1
                if part:
                    items.append({"type": "text", "text": part})
            new_message["content"] = items
        elif (
            not inserted_fallback
            and not placeholder_count
            and message.get("role") == "user"
            and isinstance(content, str)
        ):
            new_message["content"] = [
                *({"type": "image"} for _ in range(image_count)),
                {"type": "text", "text": content},
            ]
            placeholder_count = image_count
            inserted_fallback = True

        converted.append(new_message)

    if placeholder_count != image_count:
        raise ValueError(
            f"image placeholder count ({placeholder_count}) does not match "
            f"provided images ({image_count})"
        )
    return converted


def _chat_generate_hf_mm(
    processor, model, messages: list[dict], image_paths: list[str],
    cfg: MultimodalInferConfig,
) -> str:
    """HF 后端：单条多模态推理。processor = AutoProcessor。"""
    images = [Image.open(p).convert("RGB") for p in image_paths] if image_paths else None
    hf_messages = _to_hf_multimodal_messages(messages, len(image_paths))

    # 使用 processor 的 apply_chat_template（Qwen-VL / LLaVA 等均支持）
    try:
        text = processor.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # 回退：拼接 content
        text = "\n".join(str(m.get("content", "")) for m in messages)

    inputs = processor(
        text=[text], images=images, return_tensors="pt", padding=True,
    )
    target_device = first_device_of(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
    )
    if cfg.temperature and cfg.temperature > 0.0:
        gen_kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)

    # 截取新生成的 token
    input_ids = inputs.get("input_ids")
    input_len = input_ids.shape[-1] if input_ids is not None else 0
    gen_ids = out_ids[0][input_len:]
    return _post_process_text(processor.decode(gen_ids, skip_special_tokens=True))


# ===== 多模态 vLLM 推理 =====

def _chat_generate_vllm_batch_mm(
    tokenizer, llm, messages_batch: list[list[dict]],
    images_batch: list[list[str]], cfg: MultimodalInferConfig,
) -> list[str]:
    """vLLM 后端：批量多模态推理。"""
    import importlib
    SamplingParams = importlib.import_module("vllm").SamplingParams

    sampling_params = SamplingParams(
        max_tokens=cfg.max_new_tokens,
        temperature=(cfg.temperature if cfg.temperature > 0.0 else 0.0),
        top_p=(cfg.top_p if cfg.temperature > 0.0 else 1.0),
        skip_special_tokens=True,
    )
    if tokenizer.eos_token_id is not None:
        sampling_params.stop_token_ids = [tokenizer.eos_token_id]

    vllm_inputs = []
    for messages, img_paths in zip(messages_batch, images_batch):
        prompt = _build_prompt(tokenizer, messages, enable_thinking=cfg.enable_thinking)
        pil_images = [Image.open(p).convert("RGB") for p in img_paths] if img_paths else []
        entry: dict = {"prompt": prompt}
        if pil_images:
            entry["multi_modal_data"] = {"image": pil_images}
        vllm_inputs.append(entry)

    outputs = llm.generate(vllm_inputs, sampling_params, use_tqdm=False)
    return [_post_process_text(out.outputs[0].text if out.outputs else "") for out in outputs]


# ===== 核心多模态生成入口 =====

def generate_multimodal(
    samples: list[ImageTextSample],
    build_messages: Callable[[ImageTextSample], tuple[list[dict], list[str]]],
    output_path: Path,
    to_record: Callable[[ImageTextSample, str], dict],
    cfg: MultimodalInferConfig | None = None,
) -> list[dict]:
    """
    多模态统一生成入口，镜像 chat_infer.generate() 的控制流。

    参数:
        samples:        ImageTextSample 列表
        build_messages: 将样本转为 (chat_messages, image_paths) 的函数
        output_path:    输出 JSON 路径（用于中间保存）
        to_record:      将 (sample, model_output) 转为训练记录的函数
        cfg:            多模态推理配置

    返回:
        生成的训练记录列表
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logger = setup_logger(
        "generate_mm", log_dir=output_path.parent / "logs",
        file_prefix=output_path.stem,
    )

    if cfg is None:
        cfg = MultimodalInferConfig()

    # 使用 mllm_model_path，如果为空则回退到 model_path
    effective_model = cfg.mllm_model_path or cfg.model_path

    pairs: list[tuple[ImageTextSample, str]] = []
    total = len(samples)
    save_step = max(cfg.save_every, 1)

    if cfg.backend == "vllm":
        try:
            import multiprocessing as mp
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
        except Exception:
            pass

        tok, llm = load_mllm(
            model_path=effective_model,
            backend="vllm",
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            dtype=cfg.dtype,
            quantization=cfg.quantization,
            disable_log=cfg.disable_log,
        )
        bsz = max(cfg.batch_size, 1)
        next_save_at = save_step

        for start in range(0, total, bsz):
            batch = samples[start:start + bsz]
            built = [build_messages(s) for s in batch]
            msg_batch = [b[0] for b in built]
            img_batch = [b[1] for b in built]
            out_batch = _chat_generate_vllm_batch_mm(tok, llm, msg_batch, img_batch, cfg)
            pairs.extend(zip(batch, out_batch))

            done = start + len(batch)
            if done >= next_save_at:
                records = [to_record(s, o) for s, o in pairs]
                save_json(output_path, records)
                logger.info(f"[Progress] {done}/{total}，已保存 {len(records)} 条 -> {output_path}")
                next_save_at += save_step
    else:
        # HF
        proc, mdl = load_mllm(
            model_path=effective_model,
            backend="hf",
            quantization=cfg.quantization,
            dtype_gpu=cfg.dtype_gpu,
            dtype_cpu=cfg.dtype_cpu,
        )
        for idx, s in enumerate(samples, start=1):
            msgs, imgs = build_messages(s)
            out = _chat_generate_hf_mm(proc, mdl, msgs, imgs, cfg)
            pairs.append((s, out))
            if idx % save_step == 0:
                records = [to_record(s, o) for s, o in pairs]
                save_json(output_path, records)
                logger.info(f"[Progress] {idx}/{total}，已保存 {len(records)} 条 -> {output_path}")

    records = [to_record(s, o) for s, o in pairs]
    save_json(output_path, records)
    logger.info(f"[Done] 多模态生成完成，共 {len(records)} 条 -> {output_path}")
    return records
