#!/usr/bin/env bash
# run_efficiency_all.sh
# Runs PyTorch baseline + all TRT precision benchmarks sequentially.
# Usage: bash run_all_benchmarks.sh [--num_samples N] [--warmup W] [--max_new_tokens T]
# Defaults: num_samples=28, warmup=3, max_new_tokens=256

set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
NUM_SAMPLES=50
WARMUP=3
MAX_NEW_TOKENS=256
BENCH_DIR="/workspace/benchmark/efficiency"
TRT_ENGINE_BASE="/workspace/trt_engines"

# ── Parse optional overrides ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_samples)   NUM_SAMPLES="$2";   shift 2 ;;
    --warmup)        WARMUP="$2";        shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── TRT precision configs: (output_tag  engine_dir) ──────────────────────────
declare -A TRT_ENGINES=(
  [bf16]="$TRT_ENGINE_BASE/qwen2vl_2b_bf16"
  [fp8]="$TRT_ENGINE_BASE/qwen2vl_2b_fp8"
  [int8]="$TRT_ENGINE_BASE/qwen2vl_2b_int8"
  [int4]="$TRT_ENGINE_BASE/qwen2vl_2b_int4"
  [int4_awq]="$TRT_ENGINE_BASE/qwen2vl_2b_int4_awq"
)
# Run order (associative arrays are unordered in bash)  
TRT_ORDER=(bf16 fp8 int8 int4)

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }
ok()   { echo -e "\033[1;32m  ✓ $*\033[0m"; }
fail() { echo -e "\033[1;31m  ✗ $*\033[0m"; }

FAILED=()

run_cmd() {
  local label="$1"; shift
  log "START: $label"
  echo "  CMD: $*"
  if "$@"; then
    ok "$label done"
  else
    fail "$label FAILED (exit $?)"
    FAILED+=("$label")
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
cd "$BENCH_DIR"

echo "================================================================="
echo " Full Benchmark Suite"
echo " Samples: $NUM_SAMPLES | Warmup: $WARMUP | Max new tokens: $MAX_NEW_TOKENS"
echo "================================================================="

# 1. PyTorch baseline
run_cmd "baseline (bf16)" \
  python3 run_benchmark_baseline.py \
    --output_tag bf16_v1

# 2. TRT benchmarks
for PRECISION in "${TRT_ORDER[@]}"; do
  ENGINE_DIR="${TRT_ENGINES[$PRECISION]}"

  if [[ ! -d "$ENGINE_DIR" ]]; then
    fail "Engine dir not found, skipping $PRECISION: $ENGINE_DIR"
    FAILED+=("trt_$PRECISION")
    continue
  fi

  run_cmd "trt ($PRECISION)" \
    python3 run_benchmark_trt.py \
      --engine_dir "$ENGINE_DIR" \
      --precision   "$PRECISION" \
      --output_tag  "${PRECISION}_v1"
done

run_cmd "trt (int4_awq)" \
  python3 run_benchmark_trt.py \
    --engine_dir "${TRT_ENGINES[int4_awq]}"" \
    --precision   int4 \
    --output_tag int4_awq_v1

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================="
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo -e "\033[1;32m ALL BENCHMARKS PASSED\033[0m"
else
  echo -e "\033[1;31m FAILED: ${FAILED[*]}\033[0m"
fi
echo "================================================================="
echo "Results saved to:"
echo "  baseline -> $BENCH_DIR/results/efficiency/baseline/"
echo "  trt      -> $BENCH_DIR/results/efficiency/trt/"
echo "================================================================="
