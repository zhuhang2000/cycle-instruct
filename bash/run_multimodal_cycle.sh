#!/usr/bin/env bash
# ===========================================================================
# Multimodal Cycle-Instruct: 一键运行脚本
#
# 串联 Stage 0 (数据清洗) → Stage 1 (正向 VQA 生成) →
#       Stage 2 (循环验证) → Stage 3 (过滤导出)
#
# 用法:
#   bash run_multimodal_cycle.sh \
#     -i /path/to/raw_data.pdf \
#     -o /path/to/output_dir \
#     -m /path/to/mllm_model \
#     --text-model /path/to/text_llm
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

# ===== 默认参数 =====
INPUT=""
OUTPUT_DIR=""
MLLM_MODEL=""
VERIFIER_MODEL=""
TEXT_MODEL=""
BACKEND="vllm"
QUANTIZATION=""
NUM_QA=3
CYCLE_THRESHOLD=0.70
DATA_TYPE="pdf"   # pdf | web | json

# ===== 参数解析 =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input | -i)       INPUT="$2"; shift 2 ;;
    --output-dir | -o)  OUTPUT_DIR="$2"; shift 2 ;;
    --mllm-model | -m)  MLLM_MODEL="$2"; shift 2 ;;
    --verifier-model)   VERIFIER_MODEL="$2"; shift 2 ;;
    --text-model)       TEXT_MODEL="$2"; shift 2 ;;
    --backend | -bk)    BACKEND="$2"; shift 2 ;;
    --quantization | -q) QUANTIZATION="$2"; shift 2 ;;
    --num-qa | -n)      NUM_QA="$2"; shift 2 ;;
    --cycle-threshold)  CYCLE_THRESHOLD="$2"; shift 2 ;;
    --data-type | -t)   DATA_TYPE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash run_multimodal_cycle.sh [options]"
      echo ""
      echo "Options:"
      echo "  -i, --input PATH            原始数据路径 (PDF / HTML / JSON)"
      echo "  -o, --output-dir PATH        输出根目录"
      echo "  -m, --mllm-model PATH        多模态 LLM 路径"
      echo "  --verifier-model PATH        Verifier MLLM 路径 (默认同 -m)"
      echo "  --text-model PATH            文本 LLM 路径 (用于 A2Q 问题重建)"
      echo "  -bk, --backend vllm|hf       推理后端 (默认 vllm)"
      echo "  -q, --quantization Q         量化方式"
      echo "  -n, --num-qa N               每张图生成 QA 对数 (默认 3)"
      echo "  --cycle-threshold FLOAT      综合分阈值 (默认 0.70)"
      echo "  -t, --data-type pdf|web|json  输入数据类型 (默认 pdf)"
      exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 2 ;;
  esac
done

# 参数校验
if [[ -z "$INPUT" || -z "$OUTPUT_DIR" || -z "$MLLM_MODEL" ]]; then
  echo "[ERROR] --input, --output-dir, --mllm-model 均为必填参数" >&2
  exit 1
fi

TEXT_MODEL="${TEXT_MODEL:-$MLLM_MODEL}"
VERIFIER_MODEL="${VERIFIER_MODEL:-$MLLM_MODEL}"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
LOGDIR="$OUTPUT_DIR/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/pipeline_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

# =============================================
# Stage 0: 数据清洗
# =============================================
log "========== Stage 0: 数据清洗 =========="
SAMPLES_JSONL="$OUTPUT_DIR/stage0_samples.jsonl"

case "$DATA_TYPE" in
  pdf)
    log "从 PDF 提取图文对: $INPUT"
    "$PYTHON_BIN" code/data_cleaning/pdf_extractor.py \
      -i "$INPUT" \
      -o "$OUTPUT_DIR/stage0_raw" 2>&1 | tee -a "$LOGFILE"
    # pdf_extractor 输出到 stage0_raw/samples.jsonl
    cp "$OUTPUT_DIR/stage0_raw/samples.jsonl" "$SAMPLES_JSONL"
    ;;
  web)
    log "从 Web 提取图文对: $INPUT"
    "$PYTHON_BIN" code/data_cleaning/web_extractor.py \
      -i "$INPUT" \
      -o "$OUTPUT_DIR/stage0_raw" \
      --clip-filter 2>&1 | tee -a "$LOGFILE"
    cp "$OUTPUT_DIR/stage0_raw/samples.jsonl" "$SAMPLES_JSONL"
    ;;
  json)
    log "直接使用 JSON 输入: $INPUT"
    cp "$INPUT" "$SAMPLES_JSONL"
    ;;
  *)
    echo "[ERROR] 不支持的 data-type: $DATA_TYPE" >&2
    exit 1 ;;
esac

# 去重 + 质量过滤
CLEAN_JSONL="$OUTPUT_DIR/stage0_clean.jsonl"
log "去重 + 质量过滤"
"$PYTHON_BIN" code/data_cleaning/dedup_and_filter.py \
  -i "$SAMPLES_JSONL" \
  -o "$CLEAN_JSONL" 2>&1 | tee -a "$LOGFILE"

# JSONL → JSON (Stage 1 需要 JSON 输入)
CLEAN_JSON="$OUTPUT_DIR/stage0_clean.json"
"$PYTHON_BIN" -c "
import json
with open('$CLEAN_JSONL') as f:
    data = [json.loads(l) for l in f if l.strip()]
with open('$CLEAN_JSON', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f'转换 {len(data)} 条 JSONL → JSON')
" 2>&1 | tee -a "$LOGFILE"

# =============================================
# Stage 1: 正向 VQA 生成
# =============================================
log "========== Stage 1: 正向 VQA 生成 =========="
VQA_RAW="$OUTPUT_DIR/stage1_vqa_raw.json"

STAGE1_ARGS=(
  "$PYTHON_BIN" code/I2QA/generate_vqa_pairs.py
  -i "$CLEAN_JSON"
  -o "$VQA_RAW"
  -bk "$BACKEND"
  -m "$MLLM_MODEL"
  -n "$NUM_QA"
)
[[ -n "$QUANTIZATION" ]] && STAGE1_ARGS+=(-q "$QUANTIZATION")

log "Running: ${STAGE1_ARGS[*]}"
"${STAGE1_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"

# =============================================
# Stage 2: 循环一致性验证
# =============================================
log "========== Stage 2: 循环一致性验证 =========="
VQA_SCORED="$OUTPUT_DIR/stage2_vqa_scored.json"

STAGE2_ARGS=(
  "$PYTHON_BIN" code/I2QA/verify_cycle_consistency.py
  -i "$VQA_RAW"
  -o "$VQA_SCORED"
  -bk "$BACKEND"
  -m "$MLLM_MODEL"
  -vm "$VERIFIER_MODEL"
  --text-model-path "$TEXT_MODEL"
)
[[ -n "$QUANTIZATION" ]] && STAGE2_ARGS+=(-q "$QUANTIZATION")

log "Running: ${STAGE2_ARGS[*]}"
"${STAGE2_ARGS[@]}" 2>&1 | tee -a "$LOGFILE"

# =============================================
# Stage 3: 过滤导出
# =============================================
log "========== Stage 3: 过滤导出 =========="
TRAINING_DATA="$OUTPUT_DIR/training_data.json"

"$PYTHON_BIN" code/I2QA/filter_and_export.py \
  -i "$VQA_SCORED" \
  -o "$TRAINING_DATA" \
  --cycle-threshold "$CYCLE_THRESHOLD" 2>&1 | tee -a "$LOGFILE"

# =============================================
# 完成
# =============================================
log "========== Pipeline 完成 =========="
log "最终训练数据: $TRAINING_DATA"
log "日志: $LOGFILE"
