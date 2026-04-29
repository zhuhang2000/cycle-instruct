#!/usr/bin/env bash
# Iterative Cycle-Instruct multi-round orchestration.
#
# Usage:
#   bash bash/run_iterative_cycle.sh \
#       --model /path/to/Qwen3-VL-4B \
#       --seed_data seeds.json \
#       --images raw_images/ \
#       --output runs/$(date +%Y%m%d_%H%M) \
#       --max_rounds 3 \
#       --samples_per_round 500
#
# Smoke test (mocked, no GPU required):
#   CI_SKIP_HEAVY=1 bash bash/run_iterative_cycle.sh --smoke
#
set -euo pipefail

# -------- defaults --------
MODEL_PATH=""
SEED_DATA=""
IMAGES=""
OUTPUT=""
LF_DATA_DIR=""
LF_TEMPLATE="qwen3_vl_nothink"
MAX_ROUNDS=5
SAMPLES_PER_ROUND=2000
DRY_RUN=""
SMOKE=""

# -------- arg parsing --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model|--model_path|--base_model) MODEL_PATH="$2"; shift 2 ;;
    --seed_data)          SEED_DATA="$2"; shift 2 ;;
    --images)             IMAGES="$2"; shift 2 ;;
    --output)             OUTPUT="$2"; shift 2 ;;
    --llamafactory_data)  LF_DATA_DIR="$2"; shift 2 ;;
    --template|--llamafactory_template) LF_TEMPLATE="$2"; shift 2 ;;
    --max_rounds)         MAX_ROUNDS="$2"; shift 2 ;;
    --samples_per_round)  SAMPLES_PER_ROUND="$2"; shift 2 ;;
    --dry-run)            DRY_RUN="--dry_run"; shift 1 ;;
    --smoke)              SMOKE="1"; shift 1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

# -------- smoke-test presets --------
if [[ -n "$SMOKE" ]]; then
  TMP=$(mktemp -d)
  echo "[smoke] using tmp dir $TMP"
  MODEL_PATH="${MODEL_PATH:-$TMP/base_model}"
  SEED_DATA="${SEED_DATA:-$TMP/seeds.json}"
  IMAGES="${IMAGES:-$TMP/images}"
  OUTPUT="${OUTPUT:-$TMP/run}"
  mkdir -p "$MODEL_PATH" "$IMAGES" "$OUTPUT"
  echo "[]" > "$SEED_DATA"
  MAX_ROUNDS=2
  SAMPLES_PER_ROUND=10
  export CI_SKIP_HEAVY=1
fi

# -------- required args --------
for v in MODEL_PATH SEED_DATA IMAGES OUTPUT; do
  if [[ -z "${!v}" ]]; then
    echo "missing --${v,,} (got empty)"
    exit 2
  fi
done

mkdir -p "$OUTPUT"

echo "[iterative] model_path=$MODEL_PATH"
echo "[iterative] seed_data=$SEED_DATA"
echo "[iterative] images=$IMAGES"
echo "[iterative] output=$OUTPUT"
echo "[iterative] llamafactory_template=$LF_TEMPLATE"
echo "[iterative] max_rounds=$MAX_ROUNDS samples/round=$SAMPLES_PER_ROUND"

LF_ARG=()
if [[ -n "$LF_DATA_DIR" ]]; then
  LF_ARG=(--llamafactory_data_dir "$LF_DATA_DIR")
fi

python -m code.iterative.iterative_trainer \
  --model_path "$MODEL_PATH" \
  --initial_data_path "$SEED_DATA" \
  --raw_image_dir "$IMAGES" \
  --output_root "$OUTPUT" \
  --llamafactory_template "$LF_TEMPLATE" \
  --max_rounds "$MAX_ROUNDS" \
  --samples_per_round "$SAMPLES_PER_ROUND" \
  $DRY_RUN \
  "${LF_ARG[@]}"

echo "[iterative] DONE"
