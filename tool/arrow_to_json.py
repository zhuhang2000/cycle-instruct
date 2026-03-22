import argparse
import importlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 .arrow 文件转换为 JSON 文件")
    parser.add_argument("input", help="输入的 .arrow 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        help="输出 JSON 文件路径；不传则默认与输入同名 .json",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="输出为 JSONL（每行一个 JSON 对象），默认输出为 JSON 数组",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".json")

    records = load_arrow_records(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.jsonl:
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    fmt = "JSONL" if args.jsonl else "JSON"
    print(f"转换完成: {input_path} -> {output_path} ({fmt}, 共 {len(records)} 条)")


def load_arrow_records(input_path: Path) -> list[dict]:
    # 优先使用 datasets（兼容 HuggingFace 的 .arrow）
    datasets_spec = importlib.util.find_spec("datasets")
    if datasets_spec is not None:
        datasets = importlib.import_module("datasets")
        dataset = datasets.Dataset.from_file(str(input_path))
        return [record for record in dataset]

    # 兜底使用 pyarrow
    pyarrow_spec = importlib.util.find_spec("pyarrow")
    if pyarrow_spec is not None:
        pa_ipc = importlib.import_module("pyarrow.ipc")
        with input_path.open("rb") as f:
            table = pa_ipc.open_file(f).read_all()
        return table.to_pylist()

    raise RuntimeError(
        "未检测到可用依赖，请先安装 datasets 或 pyarrow，例如: pip install datasets"
    )


if __name__ == "__main__":
    main()
