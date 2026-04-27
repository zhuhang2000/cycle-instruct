import importlib
from typing import Any

torch = importlib.import_module("torch")
transformers = importlib.import_module("transformers")
AutoTokenizer = transformers.AutoTokenizer
AutoModelForCausalLM = transformers.AutoModelForCausalLM
BitsAndBytesConfig = transformers.BitsAndBytesConfig

DEVICE_GPU = "cuda:0"
_HAS_CUDA: bool | None = None


def _has_cuda() -> bool:
    global _HAS_CUDA
    if _HAS_CUDA is None:
        _HAS_CUDA = torch.cuda.is_available()
    return _HAS_CUDA


def _normalize_quantization(quantization: str | None) -> str | None:
    if quantization is None:
        return None
    q = str(quantization).strip().lower()
    if q in {"", "none", "off", "false", "no", "0"}:
        return None
    return q


def setup_torch_perf() -> None:
    if _has_cuda():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def first_device_of(model: Any) -> str:
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for _, dev in model.hf_device_map.items():
            if isinstance(dev, int):
                return f"cuda:{dev}"
            if isinstance(dev, str) and (dev.startswith("cuda") or dev == "cpu"):
                return dev
        return "cpu"
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def load_causal_lm(*, model_path: str,
                   quantization: str | None = "4bit",
                   double_quant: bool = False,
                   quant_type: str = "nf4",
                   dtype_gpu: str = "float16",
                   dtype_cpu: str = "float32"):
    """通用 HF CausalLM 加载（支持 none/8bit/4bit）。"""
    setup_torch_perf()
    q = _normalize_quantization(quantization)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }

    has_cuda = _has_cuda()
    if has_cuda and q in {"8bit", "4bit"}:
        if q == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype_gpu),
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_quant_type=quant_type,
            )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = DEVICE_GPU if has_cuda else "cpu"
        model_kwargs["torch_dtype"] = getattr(torch, dtype_gpu) if has_cuda else getattr(torch, dtype_cpu)

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
    return tokenizer, model


def load_mllm(*, model_path: str, backend: str = "hf", **kwargs):
    """
    加载多模态 LLM（Qwen-VL / LLaVA / InternVL）。

    - HF 路径:  返回 (processor, model)
    - vLLM 路径: 返回 (tokenizer, llm)，引擎已配置 limit_mm_per_prompt
    """
    if backend == "vllm":
        extra = dict(kwargs)
        extra.setdefault("limit_mm_per_prompt", {"image": 4, "video": 2})
        return load_vllm_engine(model_path=model_path, **extra)

    # HF 路径
    setup_torch_perf()
    processor = transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    q = _normalize_quantization(kwargs.get("quantization"))

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    has_cuda = _has_cuda()
    dtype_gpu = kwargs.get("dtype_gpu", "float16")
    dtype_cpu = kwargs.get("dtype_cpu", "float32")

    if has_cuda and q in {"8bit", "4bit"}:
        if q == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype_gpu),
                bnb_4bit_use_double_quant=kwargs.get("double_quant", False),
                bnb_4bit_quant_type=kwargs.get("quant_type", "nf4"),
            )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = DEVICE_GPU if has_cuda else "cpu"
        model_kwargs["torch_dtype"] = getattr(torch, dtype_gpu) if has_cuda else getattr(torch, dtype_cpu)

    try:
        model = transformers.AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs).eval()
    except (AttributeError, ValueError, KeyError):
        # 部分 MLLM（如 Qwen2-VL）未注册 AutoModelForVision2Seq，回退 CausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
    return processor, model


def load_clip(*, model_path: str = "openai/clip-vit-large-patch14-336"):
    """加载 CLIP 模型与处理器，用于图文跨模态相似度计算。"""
    setup_torch_perf()
    CLIPModel = transformers.CLIPModel
    CLIPProcessor = transformers.CLIPProcessor
    device = DEVICE_GPU if _has_cuda() else "cpu"
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path).to(device).eval()
    return processor, model


def load_vllm_engine(*, model_path: str, **kwargs):
    """加载 vLLM 引擎并返回 (tokenizer, llm)。"""
    import os

    # 避免 Linux 下默认 fork 方式导致 CUDA 在子进程中二次初始化报错。
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    try:
        LLM = importlib.import_module("vllm").LLM
    except ImportError as exc:
        raise RuntimeError("未安装 vllm，请先执行: pip install vllm") from exc

    if kwargs.get("disable_log"):
        os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

    q = _normalize_quantization(kwargs.get("quantization"))
    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": True,
        "tensor_parallel_size": max(1, kwargs.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.9),
        "dtype": kwargs.get("dtype", "auto"),
    }
    max_model_len = kwargs.get("max_model_len", 0)
    if max_model_len and max_model_len > 0:
        llm_kwargs["max_model_len"] = max_model_len
    if q:
        llm_kwargs["quantization"] = q
    limit_mm = kwargs.get("limit_mm_per_prompt")
    if limit_mm:
        llm_kwargs["limit_mm_per_prompt"] = limit_mm

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    return tokenizer, llm


