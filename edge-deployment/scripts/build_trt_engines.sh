#!/bin/bash
# build_trt_engines.sh
# Automates TRT engine conversion for Qwen2-VL-2B-Instruct.
#
# Prerequisites (run on HOST before starting container):
#   - 30GB+ system RAM required
#   - Recommend 32GB swap for Stage 3 ONNX export:
#       sudo swapoff -a
#       sudo dd if=/dev/zero of=/swapfile bs=1G count=32
#       sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
#
# Usage (inside container):
#   chmod +x /workspace/scripts/build_trt_engines.sh
#   /workspace/scripts/build_trt_engines.sh

set -euo pipefail

MODEL_PATH="/workspace/models/Qwen2-VL-2B-Instruct"
ENGINE_DIR="/workspace/trt_engines/qwen2vl"
TRTLLM_ROOT="/app/tensorrt_llm"
MULTIMODAL_BUILDER="/usr/local/lib/python3.12/dist-packages/tensorrt_llm/tools/multimodal_builder.py"

# Validate environment
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: Model not found at $MODEL_PATH"
  echo "  Download from models/README.md (Google Drive link)"
  exit 1
fi

mkdir -p "$ENGINE_DIR"/{checkpoint,llm,vision}

# ── Stage 1 ───────────────────────────────────────────────────────
echo ""
echo "================================================="
echo " Stage 1/3: Converting HF checkpoint → TRT-LLM"
echo "================================================="

python3 "$TRTLLM_ROOT/examples/models/core/qwen/convert_checkpoint.py" \
  --model_dir "$MODEL_PATH" \
  --output_dir "$ENGINE_DIR/checkpoint" \
  --dtype float16

echo "✓ Stage 1 done: $ENGINE_DIR/checkpoint/"

# ── Stage 2 ───────────────────────────────────────────────────────
echo ""
echo "================================================="
echo " Stage 2/3: Building LLM decoder engine"
echo "================================================="

trtllm-build \
  --checkpoint_dir "$ENGINE_DIR/checkpoint" \
  --output_dir "$ENGINE_DIR/llm" \
  --gemm_plugin=float16 \
  --gpt_attention_plugin=float16 \
  --max_batch_size=4 \
  --max_input_len=2048 \
  --max_seq_len=3072 \
  --max_multimodal_len=1296

echo "✓ Stage 2 done: $ENGINE_DIR/llm/"

# ── Stage 3 ───────────────────────────────────────────────────────
echo ""
echo "================================================="
echo " Stage 3/3: Building vision encoder engine"
echo " WARNING: Requires ~20GB RAM. Takes 8-15 min."
echo "================================================="

# Patch multimodal_builder.py: fp32 → fp16 to reduce peak RAM ~25GB → ~15GB
# This only affects ONNX export, not final engine precision.
# Safe: ONNX export is graph tracing only, not inference.
if grep -q "torch_dtype=torch.float32" "$MULTIMODAL_BUILDER"; then
  echo "Applying fp16 patch to multimodal_builder.py..."
  cp "$MULTIMODAL_BUILDER" "${MULTIMODAL_BUILDER}.bak"
  sed -i 's/torch_dtype=torch.float32,/torch_dtype=torch.float16,/' "$MULTIMODAL_BUILDER"
  echo "✓ Patch applied (backup: ${MULTIMODAL_BUILDER}.bak)"
else
  echo "fp16 patch already applied or not needed, skipping."
fi

python3 "$TRTLLM_ROOT/examples/models/core/multimodal/build_multimodal_engine.py" \
  --model_type qwen2_vl \
  --model_path "$MODEL_PATH" \
  --output_dir "$ENGINE_DIR/vision" \
  --max_batch_size 1

echo "✓ Stage 3 done: $ENGINE_DIR/vision/"

# ── Smoke test ────────────────────────────────────────────────────
echo ""
echo "================================================="
echo " Smoke test: verifying pipeline end-to-end"
echo "================================================="

python3 "$TRTLLM_ROOT/examples/models/core/multimodal/run.py" \
  --hf_model_dir "$MODEL_PATH" \
  --engine_dir "$ENGINE_DIR"

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "================================================="
echo " Build complete. Engine sizes:"
echo "================================================="
du -sh "$ENGINE_DIR/llm/rank0.engine"
du -sh "$ENGINE_DIR/vision/model.engine"
echo ""
echo "Next step: run TRT benchmark"
echo "  python3 /workspace/benchmark/run_benchmark_trt.py --num_samples 50 --warmup 3"
