#!/bin/bash
# run_benchmarks.sh
# Run both baseline (PyTorch) and TRT efficiency benchmarks in sequence.
#
# Usage:
#   ./run_benchmarks.sh [OPTIONS]
#
# Options:
#   --num_samples N       Number of VQAv2 samples  (default: 50)
#   --warmup W            Warmup runs               (default: 3)
#   --max_new_tokens T    Max tokens to generate    (default: 50)
#   --output_tag TAG      Tag appended to output filenames
#   --engine_dir DIR      TRT engine directory      (default: /workspace/trt_engines/qwen2vl_v2)
#   --precision PREC      TRT precision: bf16|fp16|fp8 (default: bf16)
#   --skip_baseline       Skip PyTorch baseline benchmark
#   --skip_trt            Skip TensorRT benchmark

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_SAMPLES=50
WARMUP=3
MAX_NEW_TOKENS=50
OUTPUT_TAG=""
ENGINE_DIR="/workspace/trt_engines/qwen2vl_v2"
PRECISION="bf16"
SKIP_BASELINE=0
SKIP_TRT=0

# ── Argument Parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_samples)    NUM_SAMPLES="$2";    shift 2 ;;
    --warmup)         WARMUP="$2";         shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --output_tag)     OUTPUT_TAG="$2";     shift 2 ;;
    --engine_dir)     ENGINE_DIR="$2";     shift 2 ;;
    --precision)      PRECISION="$2";      shift 2 ;;
    --skip_baseline)  SKIP_BASELINE=1;     shift ;;
    --skip_trt)       SKIP_TRT=1;          shift ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--num_samples N] [--warmup W] [--max_new_tokens T]"
      echo "          [--output_tag TAG] [--engine_dir DIR] [--precision bf16|fp16|fp8]"
      echo "          [--skip_baseline] [--skip_trt]"
      exit 1
      ;;
  esac
done

# ── cd to this script's directory so local imports (config, metrics, …) work ──
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Tag args ──────────────────────────────────────────────────────────────────
TAG_ARGS=()
if [ -n "$OUTPUT_TAG" ]; then
  TAG_ARGS=(--output_tag "$OUTPUT_TAG")
fi

# ── Print run config ──────────────────────────────────────────────────────────
echo "================================================================="
echo " Efficiency Benchmark Runner — Swift-VLM-Flow"
echo "================================================================="
echo "  Samples       : $NUM_SAMPLES"
echo "  Warmup        : $WARMUP"
echo "  Max tokens    : $MAX_NEW_TOKENS"
echo "  Output tag    : ${OUTPUT_TAG:-(none)}"
echo "  Engine dir    : $ENGINE_DIR"
echo "  TRT precision : $PRECISION"
echo "  Skip baseline : $SKIP_BASELINE"
echo "  Skip TRT      : $SKIP_TRT"
echo "================================================================="

BASELINE_EXIT=0
TRT_EXIT=0

# ── Tool 1: PyTorch Baseline ──────────────────────────────────────────────────
if [ "$SKIP_BASELINE" -eq 0 ]; then
  echo ""
  echo "================================================================="
  echo " [Tool 1] PyTorch Baseline Benchmark"
  echo "================================================================="
  python3 run_benchmark.py \
    --num_samples    "$NUM_SAMPLES" \
    --warmup         "$WARMUP" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    "${TAG_ARGS[@]}" || BASELINE_EXIT=$?

  if [ "$BASELINE_EXIT" -ne 0 ]; then
    echo "WARNING: Baseline benchmark exited with code $BASELINE_EXIT"
  fi
else
  echo ""
  echo "[Tool 1] Baseline skipped (--skip_baseline)"
fi

# ── Tool 1b: TensorRT-LLM Benchmark ──────────────────────────────────────────
if [ "$SKIP_TRT" -eq 0 ]; then
  echo ""
  echo "================================================================="
  echo " [Tool 1b] TensorRT-LLM Benchmark (precision=$PRECISION)"
  echo "================================================================="

  if [ ! -d "$ENGINE_DIR/llm" ] || [ ! -d "$ENGINE_DIR/vision" ]; then
    echo "ERROR: TRT engine not found at $ENGINE_DIR"
    echo "       Expected subdirs: $ENGINE_DIR/llm/ and $ENGINE_DIR/vision/"
    echo "       Build engines first:  scripts/build_trt_engines_baseline.sh"
    TRT_EXIT=1
  else
    python3 run_benchmark_trt.py \
      --num_samples    "$NUM_SAMPLES" \
      --warmup         "$WARMUP" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --engine_dir     "$ENGINE_DIR" \
      --precision      "$PRECISION" \
      "${TAG_ARGS[@]}" || TRT_EXIT=$?

    if [ "$TRT_EXIT" -ne 0 ]; then
      echo "WARNING: TRT benchmark exited with code $TRT_EXIT"
    fi
  fi
else
  echo ""
  echo "[Tool 1b] TRT skipped (--skip_trt)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================="
echo " Run complete"
echo "================================================================="
echo "  Baseline exit : $BASELINE_EXIT"
echo "  TRT exit      : $TRT_EXIT"
echo ""
echo "  Results written to /workspace/results/efficiency/"
echo "    baseline/  — PyTorch JSON"
echo "    trt/       — TRT JSON"
echo "================================================================="

if [ "$BASELINE_EXIT" -ne 0 ] || [ "$TRT_EXIT" -ne 0 ]; then
  exit 1
fi