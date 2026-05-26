#!/bin/bash
# build_trt_engines_baseline.sh
# Automates TRT engine conversion for Qwen2-VL-2B-Instruct.

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
  --dtype bfloat16

# baseline only, no quantification

echo "✓ Stage 1 done: $ENGINE_DIR/checkpoint/"

# ── Stage 2 ───────────────────────────────────────────────────────
echo ""
echo "================================================="
echo " Stage 2/3: Building LLM decoder engine"
echo "================================================="

trtllm-build \
  --checkpoint_dir "$ENGINE_DIR/checkpoint" \
  --output_dir "$ENGINE_DIR/llm" \
  --gemm_plugin=bfloat16 \
  --gpt_attention_plugin=bfloat16 \
  --max_batch_size=1 \
  --max_input_len=1440 \
  --max_seq_len=2048 \
  --max_multimodal_len=1024 \
  --max_num_tokens=2048

echo "✓ Stage 2 done: $ENGINE_DIR/llm/"

# Release GPU resource
sleep 5 

# ── Stage 3 ───────────────────────────────────────────────────────
echo ""
echo "================================================="
echo " Stage 3/3: Building vision encoder engine"
echo " WARNING: Requires ~20GB RAM. Takes 8-15 min."
echo "================================================="

# Convert data type into BF16 in {MULTIMODAL_BUILDER}.
if ! grep -qP "from_pretrained.*torch_dtype=torch\.bfloat16|torch_dtype=torch\.bfloat16.*from_pretrained" "$MULTIMODAL_BUILDER"; then
  echo "Applying BF16 patch to multimodal_builder.py..."
  cp "$MULTIMODAL_BUILDER" "${MULTIMODAL_BUILDER}.bak"
  sed -i '/from_pretrained/s/torch_dtype=torch\.float\(32\|16\),/torch_dtype=torch.bfloat16,/' "$MULTIMODAL_BUILDER"
  echo "✓ Patch applied (backup: ${MULTIMODAL_BUILDER}.bak)"
else
  echo "BF16 patch already applied, skipping."
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

# ── Prepare test image ─────────────────────────────────────────
mkdir -p /workspace/test_assets

# 用 Python 直接生成一張合法的測試圖，不依賴外部 URL
python3 - <<'EOF'
from PIL import Image
img = Image.new("RGB", (224, 224), color=(100, 149, 237))
img.save("/workspace/test_assets/sample.jpg")
print("✓ Test image created: /workspace/test_assets/sample.jpg")
EOF

python3 "$TRTLLM_ROOT/examples/models/core/multimodal/run.py" \
  --hf_model_dir "$MODEL_PATH" \
  --engine_dir "$ENGINE_DIR" \
  --input_text "Describe what you see in this image in one sentence." \
  --image_path /workspace/test_assets/sample.jpg

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "================================================="
echo " Build complete. Engine sizes:"
echo "================================================="
du -sh "$ENGINE_DIR/llm/rank0.engine"
du -sh "$ENGINE_DIR/vision/model.engine"
echo ""

