#!/usr/bin/env bash
# Orchestrate the SD-off (vanilla baseline) and SD-on (EAGLE3) benchmark runs.
#
# Usage:
#   conda activate sd-eagle
#   source edge-deployment/benchmark/speculative/env.sh
#   bash run_spec_all.sh [NUM_SAMPLES] [WARMUP] [MAX_NEW_TOKENS]
#
# Results land in ../../results/speculative/{base,spec}_<run_id>.json
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_SAMPLES="${1:-60}"
WARMUP="${2:-3}"
MAX_NEW="${3:-256}"

echo ">>> SD-OFF (vanilla baseline)"
python "$HERE/run_spec_trt_edge.py" --num_samples "$NUM_SAMPLES" --warmup "$WARMUP" \
    --max_new_tokens "$MAX_NEW" --no-spec_decode

echo ">>> SD-ON (EAGLE3 speculative decoding)"
python "$HERE/run_spec_trt_edge.py" --num_samples "$NUM_SAMPLES" --warmup "$WARMUP" \
    --max_new_tokens "$MAX_NEW" --spec_decode

echo ">>> Done. Generate the report with:"
echo "    python $HERE/report_spec.py"
