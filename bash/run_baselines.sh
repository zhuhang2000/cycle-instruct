#!/usr/bin/env bash
# Run every method in an ExperimentSpec (prepare -> train -> eval).
# Usage:
#   bash bash/run_baselines.sh --spec experiments/configs/main_table.yaml
#   bash bash/run_baselines.sh --smoke --spec experiments/configs/main_table.yaml
#   bash bash/run_baselines.sh --only no_filter,clip_only_0p25 --spec ...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m experiments.baselines.runner "$@"
