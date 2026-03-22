import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将包含 input/output 的 JSON 数据转换为 LlamaFactory 可用的 alpaca 格式 JSON",
        epilog=(
            "执行示例:\n"
            "  1) 基础用法（指定输入和输出）:\n"
            "     python tool/convert_to_llamafactory_json.py data/medical-train.json data/medical-train-llamafactory.json\n\n"
            "  2) 保留原始 instruction 字段:\n"
            "     python tool/convert_to_llamafactory_json.py data/medical-train.json data/medical-train-llamafactory.json --keep-original-instruction\n\n"
            "  3) 当源字段名不是 input/output 时，指定参数名称:\n"
            "     python tool/convert_to_llamafactory_json.py data/src.json data/out.json --input-key question --output-key answer\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_file", help="输入 JSON 文件路径")
    parser.add_argument("output_file", help="输出 JSON 文件路径")
    parser.add_argument(
        "--instruction",
        default="Answer this question truthfully",
        help="当样本中没有 instruction 字段时使用的默认 instruction",
    )
    parser.add_argument(
        "--input-key",
        default="input",
        help="输入字段名，默认 input",
    )
    parser.add_argument(
        "--output-key",
        default="output",
        help="输出字段名，默认 output",
    )
    parser.add_argument(
        "--keep-original-instruction",
        action="store_true",
        help="若原始样本含 instruction，则优先保留原值",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是数组（list）")

    converted = []
    skipped = 0

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            skipped += 1
            continue

        src_input = item.get(args.input_key)
        src_output = item.get(args.output_key)

        if src_input is None or src_output is None:
            skipped += 1
            continue

        instruction = args.instruction
        if args.keep_original_instruction and item.get("instruction"):
            instruction = str(item.get("instruction"))

        converted.append(
            {
                "instruction": str(instruction),
                "input": str(src_input),
                "output": str(src_output),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(
        f"转换完成: {input_path} -> {output_path} | 保留 {len(converted)} 条，跳过 {skipped} 条"
    )


if __name__ == "__main__":
    main()
