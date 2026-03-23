"""
Q2A：给定问题，生成伪答案，导出 LlamaFactory 训练集。

用法:
    python generate_pseudo_a.py -i input.json -o output.json
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

def build_messages(question: str):
    return [
        {
            "role": "system",
            "content": (
                "You are an expert answer generator. "
                "Read the given question carefully and generate ONE concise and informative answer.\n"
                "Requirements: "
                "- Provide factual, logically sound information "
                "- Avoid vague or meaningless statements "
                "- Be concise (1–3 sentences) but complete in meaning"
            ),
        },
        {
            "role": "user",
            "content": (
                "Given the following question, write a clear, accurate, and informative answer."
                " The answer should be natural, coherent, and self-contained."
                "\n\n[Question]\n" + question
            ),
        },
    ]


def to_record(question: str, answer: str) -> dict:
    return {
        "instruction": (
            "Given the following [Answer], write a natural, unambiguous question such that "
            "the only reasonable reply to the question is exactly that [Answer].\n"
            "Requirements: include key entities and either a cause/explanation or a comparison; "
            "avoid yes/no questions."
        ),
        "input": answer,
        "output": question,
        "system": "",
        "history": [],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Q2A：生成伪答案")
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

    questions = [q for q in (read_field(s, "input", "question") for s in data) if q]
    
    # 组装 InferConfig 并传入
    cfg = InferConfig(
        backend=args.backend,
        quantization=args.quantization,
        model_path=args.model_path,
    )

    generate(questions, build_messages, output_path, to_record, cfg=cfg)
