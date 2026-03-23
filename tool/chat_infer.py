import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tool.model_loader import (
    first_device_of,
    load_causal_lm,
    load_vllm_engine,
    torch,
)
from tool.logging_utils import setup_logger

@dataclass
class InferConfig:
    """推理配置，集中管理所有可调参数的默认值。"""
    model_path: str = "/workspace/models/LLM-Research/Meta-Llama-3-8B-Instruct"
    backend: str = "vllm" # vllm / hf
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    enable_thinking: bool = False
    batch_size: int = 64
    save_every: int = 50
    # --- 统一架构配置 ---
    quantization: str | None = None       # 量化方式。HF 支持: "4bit", "8bit"; vLLM 支持: "fp8", "awq", "gptq" 等
    
    # --- HF 专用扩展参数 (当 backend="hf" 且 quantization="4bit" 时生效) ---
    double_quant: bool = False
    quant_type: str = "nf4"
    dtype_gpu: str = "float16"
    dtype_cpu: str = "float32"
    
    # --- vLLM 专用扩展参数 ---
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 0
    dtype: str = "auto"
    disable_log: bool = False


# ===== 通用工具函数 =====
def _build_prompt(tokenizer, messages, *, enable_thinking: bool = False) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _post_process_text(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text.strip()


# ===== HF 推理函数 =====
def _chat_generate_hf(tokenizer, model, messages, cfg: InferConfig) -> str:
    prompt = _build_prompt(tokenizer, messages, enable_thinking=cfg.enable_thinking)
    target_device = first_device_of(model)
    inputs = tokenizer([prompt], return_tensors="pt").to(target_device)

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if cfg.temperature and cfg.temperature > 0.0:
        gen_kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)[0]

    gen_ids = out_ids[len(inputs.input_ids[0]):]
    return _post_process_text(tokenizer.decode(gen_ids, skip_special_tokens=True))


# ===== vLLM 推理函数 =====
def _chat_generate_vllm_batch(tokenizer, llm, messages_batch, cfg: InferConfig) -> list[str]:
    import importlib
    SamplingParams = importlib.import_module("vllm").SamplingParams
    
    prompts = [
        _build_prompt(tokenizer, m, enable_thinking=cfg.enable_thinking)
        for m in messages_batch
    ]
    
    sampling_params = SamplingParams(
        max_tokens=cfg.max_new_tokens,
        temperature=(cfg.temperature if cfg.temperature > 0.0 else 0.0),
        top_p=(cfg.top_p if cfg.temperature > 0.0 else 1.0),
        skip_special_tokens=True,
    )
    if tokenizer.eos_token_id is not None:
        sampling_params.stop_token_ids = [tokenizer.eos_token_id]

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    return [_post_process_text(out.outputs[0].text if out.outputs else "") for out in outputs]


# ===== 公共工具函数 =====

def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_field(sample: dict, *keys: str) -> str:
    """按优先级从 sample 中读取第一个非空字段。"""
    for k in keys:
        v = sample.get(k)
        if v:
            return str(v).strip()
    return ""


# ===== 核心生成入口 =====
def generate(
    inputs: list[str],
    build_messages: Callable[[str], list[dict]],
    output_path: Path,
    to_record: Callable[[str, str], dict],
    cfg: InferConfig | None = None,
) -> list[dict]:
    """
    统一生成入口。

    参数:
        inputs:         输入文本列表
        build_messages: 将输入文本转为 chat messages 的函数
        output_path:    输出 JSON 路径（用于中间保存）
        to_record:      将 (input, output) 转为训练记录的函数
        cfg:            推理配置，省略则使用默认值

    返回:
        生成的训练记录列表
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    # 根据输出文件路径同级创建一个 logs 目录
    logger = setup_logger("generate", log_dir=output_path.parent / "logs", file_prefix=output_path.stem)
    
    if cfg is None:
        cfg = InferConfig()

    pairs: list[tuple[str, str]] = []
    total = len(inputs)
    save_step = max(cfg.save_every, 1)

    if cfg.backend == "vllm":
        tok, llm = load_vllm_engine(
            model_path=cfg.model_path,
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
            batch = inputs[start:start + bsz]
            msg_batch = [build_messages(x) for x in batch]
            out_batch = _chat_generate_vllm_batch(tok, llm, msg_batch, cfg)
            pairs.extend(zip(batch, out_batch))

            done = start + len(batch)
            if done >= next_save_at:
                records = [to_record(i, o) for i, o in pairs]
                save_json(output_path, records)
                logger.info(f"[Progress] {done}/{total}，已保存 {len(records)} 条 -> {output_path}")
                next_save_at += save_step
    else:
        # HF
        tok, mdl = load_causal_lm(
            model_path=cfg.model_path,
            quantization=cfg.quantization,
            double_quant=cfg.double_quant,
            quant_type=cfg.quant_type,
            dtype_gpu=cfg.dtype_gpu,
            dtype_cpu=cfg.dtype_cpu,
        )
        for idx, x in enumerate(inputs, start=1):
            out = _chat_generate_hf(tok, mdl, build_messages(x), cfg)
            pairs.append((x, out))
            if idx % save_step == 0:
                records = [to_record(i, o) for i, o in pairs]
                save_json(output_path, records)
                logger.info(f"[Progress] {idx}/{total}，已保存 {len(records)} 条 -> {output_path}")

    records = [to_record(i, o) for i, o in pairs]
    save_json(output_path, records)
    logger.info(f"[Done] 生成完成，共 {len(records)} 条 -> {output_path}")
    return records
