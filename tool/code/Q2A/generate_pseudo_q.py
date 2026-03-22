import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# ===== 0) 环境与设备 =====
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# 本地模型软链接路径（你已建立的）
MODEL_PATH_A2Q = os.path.expanduser("/public/home/robertchen/zh20254227049/code/project/models/qwen3Lora_combine_A2Q_1")

DTYPE_GPU = "bfloat16"
DTYPE_CPU = "float16"  # CPU 上为省内存，若内存充足可用 "float32"


def _first_device_of(model):
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

def load_model(path):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # pad_token 兜底，避免生成时报错/警告
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",                          # 多卡自动切分
        dtype=DTYPE_GPU if torch.cuda.is_available() else DTYPE_CPU,
        trust_remote_code=True,
        # 可选：如果环境支持，能省显存/提速
        # attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).eval()

    return tok, model

# ===== 1) 通用生成函数 =====
def chat_generate(tokenizer, model, messages, *,
                  max_new_tokens=256, temperature=0.0,
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
        gen_kwargs.update(dict(do_sample=True, temperature=temperature))
    else:
        gen_kwargs.update(dict(do_sample=False))

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)[0]

    gen_ids = out_ids[len(inputs.input_ids[0]):]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # 可选：去掉 <think>... 的内容（Qwen Thinking 模式可能产生）
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    return text.strip()

def build_messages_a2q(answer: str):
    instruction = (
        "Given the following answer, write a concise question that would elicit exactly this answer."
        " The question should be natural and unambiguous."
        "\n\n[Answer]\n" + answer
    )
    return [
        {"role": "system", "content": "You are an expert question generator.\
        Read the following paragraph and generate ONE complex and meaningful question.\
            Requirements: \
            - At least 15 Chinese characters \
            - Include important entities, causes, or comparisons\
            - Avoid yes/no questions"},
        {"role": "user", "content": instruction},
    ]

def build_qa_pairs(segments):
    tok_a2q, mdl_a2q = load_model(MODEL_PATH_A2Q)
    pairs = []
    for item in segments:
        a = item["answer"]
        q_gen = chat_generate(
        tok_a2q, mdl_a2q,
        build_messages_a2q(a),
        max_new_tokens=500,
        temperature=0.0,
        enable_thinking=False)
        pairs.append({"question": q_gen, "answer": a})
    return pairs

if __name__ == "__main__":
    # 打开并读取 JSON 文件
    #[Q,A]
    with open("/public/home/robertchen/zh20254227049/code/project/code/out_data/QA_train.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    qa_pairs = build_qa_pairs(data)

    converted = []
    for item in qa_pairs:
        q_gen = item["question"]
        a = item["answer"]
        converted.append({
            "instruction":q_gen,
            "input": "",           # 原问题没有额外输入，可留空
            "output": a,
            "system": "",          # 如果无系统提示可留空
            "history": []          # 若是多轮对话，这里可填[[q1,a1],[q2,a2]]
        })
    output_path = "/public/home/robertchen/zh20254227049/code/project/LLaMA-Factory/data/Q2A_pseudo_answer_2.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)


