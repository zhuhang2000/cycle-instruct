#!/usr/bin/env bash
# Thin wrapper around ``python -m experiments.eval.runner``.
# Usage:
#   bash bash/run_eval.sh --spec experiments/configs/main_table.yaml \
#        --method ours_round5 --model-path /path/to/merged_model \
#        --output-dir runs/experiments/main_table_v1/ours_round5
#
#   bash bash/run_eval.sh --smoke --spec experiments/configs/main_table.yaml \
#        --method no_filter --output-dir /tmp/smoke_eval
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Forward --smoke untouched. Everything else is pass-through.
python -m experiments.eval.runner "$@"
