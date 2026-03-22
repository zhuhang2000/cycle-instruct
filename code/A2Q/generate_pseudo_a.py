import os
import argparse
import importlib
import json
from pathlib import Path
from typing import Any

torch = importlib.import_module("torch")
transformers = importlib.import_module("transformers")
AutoTokenizer = transformers.AutoTokenizer
AutoModelForCausalLM = transformers.AutoModelForCausalLM
BitsAndBytesConfig = transformers.BitsAndBytesConfig

# ===== 0) 环境与设备 =====
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DTYPE_GPU = "float16"  # A10 更稳妥使用 fp16
DTYPE_CPU = "float32"
DEVICE_GPU = "cuda:0"


# A10（24GB）推荐：启用 TF32 以提升吞吐
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Q2A 模型生成伪答案并导出 LlamaFactory 训练集")
    parser.add_argument(
        "--input",
        required=True,
        help="输入 JSON 路径（每项至少包含 question 字段）",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 JSON 路径（LlamaFactory alpaca 格式）",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="本地模型路径",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="每条最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度；0 为贪心")
    parser.add_argument("--top-p", type=float, default=0.9, help="采样 top-p，仅 temperature>0 时生效")
    parser.add_argument("--save-every", type=int, default=100, help="每处理 N 条自动落盘")
    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default="4bit",
        help="量化模式：none/8bit/4bit（A10 推荐 4bit）",
    )
    parser.add_argument(
        "--double-quant",
        action="store_true",
        help="4bit 模式启用 double quant",
    )
    parser.add_argument(
        "--quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
        help="4bit 量化类型",
    )
    return parser.parse_args()


def _first_device_of(model: Any) -> str:
    """在 device_map=auto 时，找到把输入张量放在哪个设备最稳。"""
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        # 取第一层或任意一层的 device（通常是嵌入层或第0层）
        for _, dev in model.hf_device_map.items():
            if isinstance(dev, int):
                return f"cuda:{dev}"
            if isinstance(dev, str):
                if dev.startswith("cuda") or dev == "cpu":
                    return dev
        return "cpu"
    # 单设备模型的回退
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"

def load_model(path, quantization: str = "4bit", double_quant: bool = False, quant_type: str = "nf4"):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # pad_token 兜底，避免生成时报错/警告
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    use_cuda = torch.cuda.is_available()
    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }

    if use_cuda and quantization in {"8bit", "4bit"}:
        if quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, DTYPE_GPU),
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_quant_type=quant_type,
            )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = DEVICE_GPU if use_cuda else "cpu"
        model_kwargs["torch_dtype"] = getattr(torch, DTYPE_GPU) if use_cuda else getattr(torch, DTYPE_CPU)

    model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs).eval()

    if tok.padding_side != "left":
        tok.padding_side = "left"

    return tok, model

# ===== 1) 通用生成函数 =====
def chat_generate(tokenizer, model, messages, *,
                  max_new_tokens=256, temperature=0.0, top_p=0.9,
                  enable_thinking=False):
    """
    messages: [{"role":"system/user/assistant", "content": "..."}]
    返回: 纯文本（会去掉 <think>... 冗余）
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # 把输入放到模型“第一层所在的设备”
    target_device = _first_device_of(model)
    inputs = tokenizer([prompt], return_tensors="pt").to(target_device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if temperature and temperature > 0.0:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        # 显式置空，避免 transformers 对无效采样参数给出告警
        gen_kwargs.update(dict(do_sample=False, temperature=None, top_p=None))

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)[0]

    gen_ids = out_ids[len(inputs.input_ids[0]):]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # 可选：去掉 <think>... 的内容（Qwen Thinking 模式可能产生）
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    return text.strip()

def build_messages_q2a(question: str):
    instruction = (
        "Given the following question, write a clear, accurate, and informative answer."
        " The answer should be natural, coherent, and self-contained."
        "\n\n[Question]\n" + question
    )
    return [
        {
            "role": "system",
            "content": "You are an expert answer generator. "
                       "Read the given question carefully and generate ONE concise and informative answer.\n"
                       "Requirements: "
                       "- Provide factual, logically sound information "
                       "- Avoid vague or meaningless statements "
                       "- Be concise (1–3 sentences) but complete in meaning",
        },
        {"role": "user", "content": instruction},
    ]


def to_lf_record(question: str, answer: str) -> dict:
    return {
        "instruction": (
            "请根据以下【答案】生成一个自然且不含歧义的问题，"
            "使得该问题的唯一合理回答就是该【答案】。\n"
            "要求：包含重要实体、原因或比较，避免是/否问句。"
        ),
        "input": answer,
        "output": question,
        "system": "",
        "history": [],
    }


def save_json(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tok, mdl = load_model(
        args.model_path,
        quantization=args.quantization,
        double_quant=args.double_quant,
        quant_type=args.quant_type,
    )

    converted: list[dict] = []
    total = len(data)
    for idx, s in enumerate(data, start=1):
        q = str(s.get("input", "")).strip()
        if not q:
            continue

        a_gen = chat_generate(
            tok,
            mdl,
            build_messages_q2a(q),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=False,
        )
        converted.append(to_lf_record(q, a_gen))

        if idx % max(args.save_every, 1) == 0:
            save_json(output_path, converted)
            print(f"[Progress] {idx}/{total}，当前已保存 {len(converted)} 条 -> {output_path}")

    save_json(output_path, converted)
    print(f"[Done] 生成完成，共 {len(converted)} 条 -> {output_path}")


