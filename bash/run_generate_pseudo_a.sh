#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT="/workspace/cycle-instruct/origin_qa/test.json"
OUTPUT="/workspace/cycle-instruct/LlamaFactory/data/A2Q_pseudo_question_1.json"
MODEL_PATH="/workspace/models/LLM-Research/Meta-Llama-3-8B-Instruct"
MAX_NEW_TOKENS=256
TEMPERATURE=0.0
TOP_P=0.9
SAVE_EVERY=50
QUANTIZATION="4bit"    # none | 8bit | 4bit
DOUBLE_QUANT=false
QUANT_TYPE="nf4"       # nf4 | fp4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT="$2"; shift 2 ;;
    --output)
      OUTPUT="$2"; shift 2 ;;
    --model-path)
      MODEL_PATH="$2"; shift 2 ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature)
      TEMPERATURE="$2"; shift 2 ;;
    --top-p)
      TOP_P="$2"; shift 2 ;;
    --save-every)
      SAVE_EVERY="$2"; shift 2 ;;
    --quantization)
      QUANTIZATION="$2"; shift 2 ;;
    --double-quant)
      DOUBLE_QUANT=true; shift ;;
    --quant-type)
      QUANT_TYPE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash tool/run_generate_pseudo_a.sh [--input PATH] [--output PATH] [--model-path PATH] [--max-new-tokens N] [--temperature T] [--top-p P] [--save-every N] [--quantization none|8bit|4bit] [--double-quant] [--quant-type nf4|fp4]"
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

CMD=("$PYTHON_BIN" /workspace/cycle-instruct/code/A2Q/generate_pseudo_a.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --model-path "$MODEL_PATH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --save-every "$SAVE_EVERY" \
  --quantization "$QUANTIZATION" \
  --quant-type "$QUANT_TYPE")

if [[ "$DOUBLE_QUANT" == true ]]; then
  CMD+=(--double-quant)
fi

echo "[INFO] Running: ${CMD[*]}" | tee "$LOGFILE"
"${CMD[@]}" 2>&1 | tee -a "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]:-0}
if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[INFO] Completed successfully. Output -> $OUTPUT" | tee -a "$LOGFILE"
else
  echo "[ERROR] Exit code $EXIT_CODE" | tee -a "$LOGFILE"
fi

exit $EXIT_CODE
