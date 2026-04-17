#!/usr/bin/env bash
# Iterative Cycle-Instruct multi-round orchestration.
#
# Usage:
#   bash bash/run_iterative_cycle.sh \
#       --base_model /path/to/Qwen3-VL-4B \
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
BASE_MODEL=""
SEED_DATA=""
IMAGES=""
OUTPUT=""
LF_DATA_DIR=""
MAX_ROUNDS=5
SAMPLES_PER_ROUND=2000
DRY_RUN=""
SMOKE=""

# -------- arg parsing --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_model)         BASE_MODEL="$2"; shift 2 ;;
    --seed_data)          SEED_DATA="$2"; shift 2 ;;
    --images)             IMAGES="$2"; shift 2 ;;
    --output)             OUTPUT="$2"; shift 2 ;;
    --llamafactory_data)  LF_DATA_DIR="$2"; shift 2 ;;
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
  BASE_MODEL="${BASE_MODEL:-$TMP/base_model}"
  SEED_DATA="${SEED_DATA:-$TMP/seeds.json}"
  IMAGES="${IMAGES:-$TMP/images}"
  OUTPUT="${OUTPUT:-$TMP/run}"
  mkdir -p "$BASE_MODEL" "$IMAGES" "$OUTPUT"
  echo "[]" > "$SEED_DATA"
  MAX_ROUNDS=2
  SAMPLES_PER_ROUND=10
  export CI_SKIP_HEAVY=1
fi

# -------- required args --------
for v in BASE_MODEL SEED_DATA IMAGES OUTPUT; do
  if [[ -z "${!v}" ]]; then
    echo "missing --${v,,} (got empty)"
    exit 2
  fi
done

mkdir -p "$OUTPUT"

echo "[iterative] base_model=$BASE_MODEL"
echo "[iterative] seed_data=$SEED_DATA"
echo "[iterative] images=$IMAGES"
echo "[iterative] output=$OUTPUT"
echo "[iterative] max_rounds=$MAX_ROUNDS samples/round=$SAMPLES_PER_ROUND"

LF_ARG=()
if [[ -n "$LF_DATA_DIR" ]]; then
  LF_ARG=(--llamafactory_data_dir "$LF_DATA_DIR")
fi

python -m code.iterative.iterative_trainer \
  --base_model_path "$BASE_MODEL" \
  --initial_data_path "$SEED_DATA" \
  --raw_image_dir "$IMAGES" \
  --output_root "$OUTPUT" \
  --max_rounds "$MAX_ROUNDS" \
  --samples_per_round "$SAMPLES_PER_ROUND" \
  $DRY_RUN \
  "${LF_ARG[@]}"

echo "[iterative] DONE"
