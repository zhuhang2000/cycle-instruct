#!/usr/bin/env bash

# ===== 可调参数 =====
MODEL_NAME_OR_PATH="/public/home/robertchen/zh20254227049/code/project/models/qwen3Lora_combine_A2Q_1"
DATA_PATH="/public/home/robertchen/zh20254227049/code/project/LLaMA-Factory/data"
OUTPUT_DIR="/public/home/robertchen/zh20254227049/code/project/qwen3Lora/train_A2Q_2"
MAX_LEN=2048
BATCH_SIZE=1
GRAD_ACC=8
LR=1e-4
EPOCHS=3
USE_LORA=true
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"

# ===== 启动训练 =====
python -m llamafactory.cli train \
  --stage sft \
  --do_train True \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --dataset_dir ${DATA_PATH} \
  --dataset FT_A2Q \
  --output_dir "${OUTPUT_DIR}" \
  --template "chatml" \
  --cutoff_len ${MAX_LEN} \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --learning_rate ${LR} \
  --num_train_epochs ${EPOCHS} \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --logging_steps 100 \
  --save_steps 100 \
  --bf16 True \
  --plot_loss True \
  --include_num_input_tokens_seen True \
  --finetuning_type lora \
  --lora_rank ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_target all \
  --lora_dropout ${LORA_DROPOUT} \
