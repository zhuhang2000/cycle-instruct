#!/usr/bin/env bash
# Aggregate results for one experiment directory into paper_tables/.
# Usage:
#   bash bash/run_analysis.sh runs/experiments/main_table_v1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EXP_DIR="${1:?usage: run_analysis.sh <experiment_dir>}"
OUT_DIR="${2:-$EXP_DIR/paper_tables}"

python -m experiments.analysis.aggregate_tables "$EXP_DIR" --output-dir "$OUT_DIR"
python -m experiments.analysis.efficiency_report "$EXP_DIR" --output-dir "$OUT_DIR" || true
python -m experiments.analysis.ablation_table  "$EXP_DIR" --output-dir "$OUT_DIR" || true

echo "wrote tables to $OUT_DIR"
