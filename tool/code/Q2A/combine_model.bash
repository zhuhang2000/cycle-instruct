#!/bin/bash
set -e  # 一旦出现错误就退出

MODEL_NAME_OR_PATH="/public/home/robertchen/zh20254227049/code/project/models/qwen3Lora_combine_Q2A_1"
ADAPTER_PATH="/public/home/robertchen/zh20254227049/code/project/qwen3Lora/train_Q2A_2"
EXPORT_DIR="/public/home/robertchen/zh20254227049/code/project/models/qwen3Lora_combine_Q2A_2"


# 保证临时和目标目录存在
mkdir -p "$(dirname $EXPORT_DIR)"

echo "[INFO] 开始导出模型..."
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --adapter_name_or_path "${ADAPTER_PATH}" \
  --template qwen3 \
  --finetuning_type lora \
  --export_dir "${EXPORT_DIR}" \
  --export_size 2 \
  --export_legacy_format False
