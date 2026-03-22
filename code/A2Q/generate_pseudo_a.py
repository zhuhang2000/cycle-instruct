import os
import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_project_root_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "tool").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_ensure_project_root_on_path()

from tool.model_loader import first_device_of, load_causal_lm, torch

# ===== 0) 环境与设备 =====
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    parser.add_argument("--max-new-tokens", type=int, default=256, help="每条最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="生成温度；0 为贪心")
    parser.add_argument("--top-p", type=float, default=0.9, help="采样 top-p，仅 temperature>0 时生效")
    parser.add_argument("--save-every", type=int, default=100, help="每处理 N 条自动落盘")
    return parser.parse_args()


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
    target_device = first_device_of(model)
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
            "Given the following [Answer], write a natural, unambiguous question such that the only reasonable reply to the question is exactly that [Answer].\n"
            "Requirements: include key entities and either a cause/explanation or a comparison; avoid yes/no questions."
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

    tok, mdl = load_causal_lm()

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


