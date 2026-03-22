import importlib
from typing import Any


torch = importlib.import_module("torch")
transformers = importlib.import_module("transformers")
AutoTokenizer = transformers.AutoTokenizer
AutoModelForCausalLM = transformers.AutoModelForCausalLM
BitsAndBytesConfig = transformers.BitsAndBytesConfig


# 统一模型加载参数（调用方无需重复传参）
MODEL_PATH = "/workspace/models/LLM-Research/Meta-Llama-3-8B-Instruct"
QUANTIZATION = "4bit"
DOUBLE_QUANT = False
QUANT_TYPE = "nf4"
DTYPE_GPU = "float16"
DTYPE_CPU = "float32"
DEVICE_GPU = "cuda:0"


def setup_torch_perf() -> None:
    """针对 A10 等 GPU 的通用性能开关。"""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def first_device_of(model: Any) -> str:
    """在 device_map=auto 时，找到输入张量应放置的设备。"""
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


def load_causal_lm():
    """通用 CausalLM 加载（支持 none/8bit/4bit）。"""
    setup_torch_perf()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available()
    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }

    if use_cuda and QUANTIZATION in {"8bit", "4bit"}:
        if QUANTIZATION == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, DTYPE_GPU),
                bnb_4bit_use_double_quant=DOUBLE_QUANT,
                bnb_4bit_quant_type=QUANT_TYPE,
            )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = DEVICE_GPU if use_cuda else "cpu"
        model_kwargs["torch_dtype"] = getattr(torch, DTYPE_GPU) if use_cuda else getattr(torch, DTYPE_CPU)

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs).eval()
    return tokenizer, model
