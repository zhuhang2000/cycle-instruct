"""
A2Q：给定答案，生成伪问题，导出 LlamaFactory 训练集。

用法:
    python generate_pseudo_q.py -i input.json -o output.json
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

from tool.chat_infer import generate, read_field, InferConfig


# ===== 业务逻辑 =====

def build_messages(answer: str):
    return [
        {
            "role": "system",
            "content": (
                "You are an expert question generator. "
                "Read the following paragraph and generate ONE complex and meaningful question. "
                "Requirements: "
                "- At least 15 Chinese characters "
                "- Include important entities, causes, or comparisons "
                "- Avoid yes/no questions"
            ),
        },
        {
            "role": "user",
            "content": (
                "Given the following answer, write a concise question that would elicit exactly this answer."
                " The question should be natural and unambiguous."
                "\n\n[Answer]\n" + answer
            ),
        },
    ]


def to_record(answer: str, question: str) -> dict:
    return {
        "instruction": question,
        "input": "",
        "output": answer,
        "system": "",
        "history": [],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A2Q：生成伪问题")
    parser.add_argument("-i", "--input", required=True, help="输入 JSON 路径")
    parser.add_argument("-o", "--output", required=True, help="输出 JSON 路径")
    
    # 允许命令行指定关键的推理配置
    parser.add_argument("-bk","--backend", type=str, default="vllm", choices=["vllm", "hf"], help="推理后端")
    parser.add_argument("-q","--quantization", type=str, default=None, help="量化方式 (hf: 4bit/8bit, vllm: fp8/awq 等)")
    parser.add_argument("-m","--model-path", type=str, default="/workspace/models/LLM-Research/Meta-Llama-3-8B-Instruct", help="模型路径")
    
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    answers = [a for a in (read_field(s, "answer", "output", "input") for s in data) if a]
    
    # 组装 InferConfig 并传入
    cfg = InferConfig(
        backend=args.backend,
        quantization=args.quantization,
        model_path=args.model_path,
    )

    generate(answers, build_messages, output_path, to_record, cfg=cfg)
