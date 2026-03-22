from datasets import Dataset

# 路径变得非常清爽
arrow_file_path = "/workspace/my_data/oasst1-train.arrow"
dataset = Dataset.from_file(arrow_file_path)

print(f"成功加载 OASST-1，共 {len(dataset)} 条对话记录！")