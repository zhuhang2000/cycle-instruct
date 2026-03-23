#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

# bash run_generate_pseudo_q.sh -i /workspace/cycle-instruct/origin_qa/test.json -o /workspace/cycle-instruct/LlamaFactory/data/Q2A_pseudo_answer_vllm.json -bk vllm -q fp8 -m /workspace/models/LLM-Research/Meta-Llama-3-8B-Instruct

INPUT="/workspace/cycle-instruct/origin_qa/test.json"
OUTPUT="/workspace/cycle-instruct/LlamaFactory/data/Q2A_pseudo_answer_vllm.json"
BACKEND="vllm"         # vllm | hf
QUANTIZATION="fp8"     # vllm: fp8/awq/gptq/none, hf: 4bit/8bit/none
MODEL_PATH="/workspace/models/LLM-Research/Meta-Llama-3-8B-Instruct"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input | -i)
      INPUT="$2"; shift 2 ;;
    --output | -o)
      OUTPUT="$2"; shift 2 ;;
    --backend | -bk)
      BACKEND="$2"; shift 2 ;;
    --quantization | -q)
      QUANTIZATION="$2"; shift 2 ;;
    --model-path | -m)
      MODEL_PATH="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash run_generate_pseudo_q.sh [--input | -i PATH] [--output | -o PATH] [--backend | -bk vllm|hf] [--quantization | -q Q] [--model-path | -m PATH]"
      exit 0 ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 2 ;;
  esac
done

# 日志
LOGDIR="$(dirname "$OUTPUT")/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/generate_$(date +%Y%m%d_%H%M%S).log"

CMD=("$PYTHON_BIN" /workspace/cycle-instruct/code/Q2A/generate_pseudo_q.py \
  -i "$INPUT" \
  -o "$OUTPUT" \
  -bk "$BACKEND" \
  -q "$QUANTIZATION" \
  -m "$MODEL_PATH")

echo "[INFO] Running: ${CMD[*]}" | tee "$LOGFILE"
"${CMD[@]}" 2>&1 | tee -a "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]:-0}
if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[INFO] Completed successfully. Output -> $OUTPUT" | tee -a "$LOGFILE"
else
  echo "[ERROR] Exit code $EXIT_CODE" | tee -a "$LOGFILE"
fi

exit $EXIT_CODE
