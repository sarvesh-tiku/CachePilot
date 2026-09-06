#!/usr/bin/env bash
# Run one workload under every routing policy with an identical seed, then compare.
#   scripts/run_benchmark.sh [workload] [requests] [seed]
set -euo pipefail
cd "$(dirname "$0")/.."

WORKLOAD="${1:-shared_prefix_heavy}"
REQUESTS="${2:-1000}"
SEED="${3:-42}"
OUT_DIR="${OUT_DIR:-benchmark/results}"
PYTHON="${PYTHON:-python}"
# Works with or without an editable install of the backend package.
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}"

for policy in round_robin least_loaded kv_aware; do
  "$PYTHON" -m benchmark.runner \
    --workload "$WORKLOAD" --policy "$policy" \
    --requests "$REQUESTS" --seed "$SEED" --out-dir "$OUT_DIR" --quiet
done

"$PYTHON" -m benchmark.analysis "$OUT_DIR/${WORKLOAD}"_*_seed"${SEED}".json
