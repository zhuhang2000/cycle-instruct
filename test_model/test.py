import argparse
import re
import sys
from pathlib import Path

from datasets import Dataset



def _ensure_project_root_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "tool").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_ensure_project_root_on_path()

from tool.model_loader import first_device_of, load_causal_lm, torch
from tool.chat_infer import InferConfig
from tool.logging_utils import setup_logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用本地模型评测 MMLU arrow 测试集",
        epilog=(
            "日志路径说明:\n"
            "  1) 不传 --log-file: 自动生成到 test_model/logs/mmlu_eval_YYYYMMDD_HHMMSS.log\n"
            "  2) 传相对路径: 相对当前执行目录，例如 --log-file logs/run1.log\n"
            "  3) 传绝对路径: 直接写入指定位置，例如 --log-file /workspace/logs/mmlu_eval.log\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--file-path", required=True, help="mmlu-test.arrow 路径")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--print-samples", type=int, default=2, help="打印前 N 条样例")
    parser.add_argument(
        "--log-file",
        default="",
        help=(
            "日志文件路径。为空时自动生成: test_model/logs/mmlu_eval_YYYYMMDD_HHMMSS.log；"
            "可传相对路径或绝对路径"
        ),
    )
    return parser.parse_args()


def build_prompt(question: str, choices: list[str]) -> str:
    return (
        "你是一个严谨的选择题助手。请从 A/B/C/D 中选择唯一正确选项。"
        "只输出一个大写字母，不要输出其它内容。\n\n"
        f"Question: {question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )


def generate_answer_letter(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    target_device = first_device_of(model)
    inputs = tokenizer([chat_prompt], return_tensors="pt").to(target_device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        gen_kwargs.update({"do_sample": False, "temperature": None, "top_p": None})

    with torch.inference_mode():
        out_ids = model.generate(**inputs, **gen_kwargs)[0]
    gen_ids = out_ids[len(inputs.input_ids[0]):]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # 提取第一个 A/B/C/D
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    for ch in text.upper():
        if ch in "ABCD":
            return ch
    return ""


# 用于将数字索引映射为选项字母
idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

def main() -> None:
    args = parse_args()
    logger = setup_logger(
        "mmlu_eval",
        args.log_file or None,
        log_dir=Path("test_model") / "logs",
        file_prefix="mmlu_eval",
    )

    try:
        test_dataset = Dataset.from_file(args.file_path)
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise SystemExit(1)

    tokenizer, model = load_causal_lm(model_path=InferConfig.model_path)

    correct_count = 0
    total_count = len(test_dataset)

    logger.info(f"成功加载测试集，共 {total_count} 道题目。开始测试...")

    for i, item in enumerate(test_dataset):
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        correct_answer = idx_to_letter[answer_idx]
    
        prompt = build_prompt(question, choices)
        predicted_letter = generate_answer_letter(
            tokenizer,
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    
        if predicted_letter == correct_answer:
            correct_count += 1

        if i < args.print_samples:
            logger.info(f"--- 题目 {i + 1} ---")
            logger.info(prompt)
            logger.info(f"模型预测: {predicted_letter}, 正确答案: {correct_answer}")

    accuracy = (correct_count / total_count) * 100 if total_count else 0.0
    logger.info("-" * 30)
    logger.info("测试完成！")
    logger.info(f"总题数: {total_count}")
    logger.info(f"答对题数: {correct_count}")
    logger.info(f"准确率 (Accuracy): {accuracy:.2f}%")


if __name__ == "__main__":
    main()