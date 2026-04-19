#!/usr/bin/env bash
# Run the intrinsic-metric suite over a VQA JSONL pool.
# Usage:
#   bash bash/run_intrinsic.sh --input runs/round_3/filtered.jsonl --out report/round_3
#   bash bash/run_intrinsic.sh --smoke --input tests/fixtures/tiny_vqa.jsonl --out /tmp/report
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m experiments.intrinsic.report "$@"
