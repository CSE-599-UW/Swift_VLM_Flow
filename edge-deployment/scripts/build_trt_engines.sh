#!/bin/bash
# build_trt_engines.sh
# Automates TRT engine conversion for Qwen VL model family.
# Supports multiple models and quantization modes via two internal pipelines.
#
# Usage:
#   bash build_trt_engines.sh --model qwen2vl_2b --quant bf16
#   bash build_trt_engines.sh --model qwen2vl_2b --quant int8
#   bash build_trt_engines.sh --model qwen2vl_2b --quant int4
#   bash build_trt_engines.sh --model qwen2vl_2b --quant smoothquant
#   bash build_trt_engines.sh --model qwen2vl_2b --quant fp8
#   bash build_trt_engines.sh --model qwen2vl_2b --quant int4_awq
#
# Supported --model values:
#   qwen2vl_2b   → Qwen2-VL-2B-Instruct    (✅ 穩定)
#   qwen2vl_7b   → Qwen2-VL-7B-Instruct    (✅ 穩定)
#   qwen25vl_7b  → Qwen2.5-VL-7B-Instruct  (⚠️  需要 TRT-LLM v1.3+)
#   qwen3vl_2b   → Qwen3-VL-2B-Instruct    (⚠️  需要 TRT-LLM v1.3+ nightly，有已知 bug)
#
# Supported --quant values:
#   ── Pipeline A: convert_checkpoint.py（不需要 calibration）──
#   bf16        → BFloat16 baseline
#   int8        → W8A16，INT8 weight-only
#   int4        → W4A16，INT4 weight-only
#   smoothquant → W8A8，weight + activation 同時量化（需要 calibration data）
#
#   ── Pipeline B: quantize.py via ModelOpt（需要 calibration）──
#   fp8         → W8A8 FP8，需要 Ada/Hopper/Blackwell GPU (RTX 5060 Ti ✅)
#   int4_awq    → W4A16 AWQ，精度最佳的 4-bit 選項

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────
MODEL="qwen2vl_2b"
QUANT="bf16"

# ── Parse args ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --quant)
      QUANT="$2"
      shift 2
      ;;
    --help|-h)
      sed -n '2,30p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      echo "Run '$0 --help' for usage."
      exit 1
      ;;
  esac
done

# ── Model registry ────────────────────────────────────────────────
case "$MODEL" in
  qwen2vl_2b)
    HF_MODEL_NAME="Qwen2-VL-2B-Instruct"
    MULTIMODAL_TYPE="qwen2_vl"
    TRT_SUPPORT="stable"
    ;;
  qwen2vl_7b)
    HF_MODEL_NAME="Qwen2-VL-7B-Instruct"
    MULTIMODAL_TYPE="qwen2_vl"
    TRT_SUPPORT="stable"
    ;;
  qwen25vl_7b)
    HF_MODEL_NAME="Qwen2.5-VL-7B-Instruct"
    MULTIMODAL_TYPE="qwen2_5_vl"
    TRT_SUPPORT="requires_v13"
    ;;
  qwen3vl_2b)
    HF_MODEL_NAME="Qwen3-VL-2B-Instruct"
    MULTIMODAL_TYPE="qwen3_vl"
    TRT_SUPPORT="experimental"
    ;;
  *)
    echo "ERROR: Unsupported --model value: '$MODEL'"
    echo "  Valid options: qwen2vl_2b | qwen2vl_7b | qwen25vl_7b | qwen3vl_2b"
    exit 1
    ;;
esac

# ── Quant mode validation ─────────────────────────────────────────
case "$QUANT" in
  bf16|int8|int4|smoothquant|fp8|int4_awq) ;;
  *)
    echo "ERROR: Unsupported --quant value: '$QUANT'"
    echo "  Pipeline A (no calibration): bf16 | int8 | int4 | smoothquant"
    echo "  Pipeline B (needs calibration): fp8 | int4_awq"
    exit 1
    ;;
esac

# ── Determine pipeline ────────────────────────────────────────────
# Pipeline A: convert_checkpoint.py（原生，不需要 calibration）
# Pipeline B: quantize.py via ModelOpt（需要 calibration data）
case "$QUANT" in
  bf16|int8|int4|smoothquant)
    PIPELINE="A"
    ;;
  fp8|int4_awq)
    PIPELINE="B"
    ;;
esac

# ── Support warnings ──────────────────────────────────────────────
case "$TRT_SUPPORT" in
  requires_v13)
    echo ""
    echo "⚠️  WARNING: $HF_MODEL_NAME 需要 TRT-LLM v1.3+。"
    echo "   確認版本：python3 -c \"import tensorrt_llm; print(tensorrt_llm.__version__)\""
    echo "   繼續執行中... (Ctrl+C 可取消)"
    sleep 5
    ;;
  experimental)
    echo ""
    echo "⚠️  WARNING: $HF_MODEL_NAME 為實驗性支援，有已知 bug。"
    echo "   繼續執行中... (Ctrl+C 可取消)"
    sleep 5
    ;;
esac

# ── Paths ─────────────────────────────────────────────────────────
MODEL_PATH="/workspace/models/${HF_MODEL_NAME}"
ENGINE_DIR="/workspace/trt_engines/${MODEL}_${QUANT}"
TRTLLM_ROOT="/app/tensorrt_llm"
MULTIMODAL_BUILDER="/usr/local/lib/python3.12/dist-packages/tensorrt_llm/tools/multimodal_builder.py"
QUANTIZE_SCRIPT="$TRTLLM_ROOT/examples/quantization/quantize.py"

echo ""
echo "================================================="
echo " Model             : $HF_MODEL_NAME"
echo " Quantization      : $QUANT  (Pipeline $PIPELINE)"
echo " Engine output dir : $ENGINE_DIR"
echo "================================================="

# ── Validate environment ──────────────────────────────────────────
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: Model not found at $MODEL_PATH"
  exit 1
fi

if [ "$PIPELINE" = "B" ] && [ ! -f "$QUANTIZE_SCRIPT" ]; then
  echo "ERROR: quantize.py not found at $QUANTIZE_SCRIPT"
  echo "  This script is required for fp8 and int4_awq quantization."
  exit 1
fi

mkdir -p "$ENGINE_DIR"/{checkpoint,llm,vision}

# ══════════════════════════════════════════════════════════════════
#  STAGE 1 — Checkpoint conversion
# ══════════════════════════════════════════════════════════════════
echo ""
echo "================================================="
echo " Stage 1/3: Converting HF checkpoint → TRT-LLM"
echo " Pipeline $PIPELINE / quant=$QUANT"
echo "================================================="

case "$QUANT" in

  # ── Pipeline A ────────────────────────────────────────────────
  bf16)
    # Weight: BF16 / Activation: BF16
    PLUGIN_DTYPE="bfloat16"
    VISION_DTYPE_PATCH="bfloat16"
    python3 "$TRTLLM_ROOT/examples/models/core/qwen/convert_checkpoint.py" \
      --model_dir  "$MODEL_PATH" \
      --output_dir "$ENGINE_DIR/checkpoint" \
      --dtype bfloat16
    ;;

  int8)
    # Weight: INT8 / Activation: FP16
    PLUGIN_DTYPE="float16"
    VISION_DTYPE_PATCH="float16"
    python3 "$TRTLLM_ROOT/examples/models/core/qwen/convert_checkpoint.py" \
      --model_dir  "$MODEL_PATH" \
      --output_dir "$ENGINE_DIR/checkpoint" \
      --dtype float16 \
      --use_weight_only \
      --weight_only_precision int8
    ;;

  int4)
    # Weight: INT4 / Activation: FP16
    PLUGIN_DTYPE="float16"
    VISION_DTYPE_PATCH="float16"
    python3 "$TRTLLM_ROOT/examples/models/core/qwen/convert_checkpoint.py" \
      --model_dir  "$MODEL_PATH" \
      --output_dir "$ENGINE_DIR/checkpoint" \
      --dtype float16 \
      --use_weight_only \
      --weight_only_precision int4
    ;;

  smoothquant)
    # Weight: INT8 / Activation: INT8 (W8A8)
    # --smoothquant 0.5: 遷移強度，越大越多量化負擔移到 weight 側（範圍 0~1）
    # --per_token --per_channel: 精度最佳的 SmoothQuant 組合
    PLUGIN_DTYPE="float16"
    VISION_DTYPE_PATCH="float16"
    python3 "$TRTLLM_ROOT/examples/models/core/qwen/convert_checkpoint.py" \
      --model_dir  "$MODEL_PATH" \
      --output_dir "$ENGINE_DIR/checkpoint" \
      --dtype float16 \
      --smoothquant 0.5 \
      --per_token \
      --per_channel
    ;;

  # ── Pipeline B ────────────────────────────────────────────────
  fp8)
    # Weight: FP8 / Activation: FP8 (W8A8)
    # 需要 Ada Lovelace (RTX 4000) 或更新 GPU → RTX 5060 Ti ✅
    # --calib_size 512: calibration 用的樣本數，越大越準但越慢
    PLUGIN_DTYPE="float16"
    VISION_DTYPE_PATCH="float16"
    python3 "$QUANTIZE_SCRIPT" \
      --model_dir  "$MODEL_PATH" \
      --output_dir "$ENGINE_DIR/checkpoint" \
      --dtype float16 \
      --qformat fp8 \
      --kv_cache_dtype fp8 \
      --calib_size 512
    ;;

  int4_awq)
    # Weight: INT4 per-group / Activation: FP16 (W4A16)
    # AWQ: calibration 時保留 salient weights，精度比純 int4 好
    # --awq_block_size 128: per-group 的 group size，128 是 Qwen 推薦值
    # --calib_size 32: AWQ 需要的 calibration 樣本數較少
    PLUGIN_DTYPE="float16"
    VISION_DTYPE_PATCH="float16"
    python3 "$QUANTIZE_SCRIPT" \
      --model_dir  "$MODEL_PATH" \
      --output_dir "$ENGINE_DIR/checkpoint" \
      --dtype float16 \
      --qformat int4_awq \
      --awq_block_size 128 \
      --calib_size 32
    ;;

esac

echo "✓ Stage 1 done: $ENGINE_DIR/checkpoint/"

# ══════════════════════════════════════════════════════════════════
#  STAGE 2 — Build LLM decoder engine
# ══════════════════════════════════════════════════════════════════
echo ""
echo "================================================="
echo " Stage 2/3: Building LLM decoder engine"
echo " plugin_dtype=$PLUGIN_DTYPE"
echo "================================================="

# fp8 需要額外的 context fmha flag
FP8_FLAG=""
if [ "$QUANT" = "fp8" ]; then
  FP8_FLAG="--use_fp8_context_fmha=enable"
fi

# shellcheck disable=SC2086
trtllm-build \
  --checkpoint_dir "$ENGINE_DIR/checkpoint" \
  --output_dir     "$ENGINE_DIR/llm" \
  --gemm_plugin="$PLUGIN_DTYPE" \
  --gpt_attention_plugin="$PLUGIN_DTYPE" \
  --max_batch_size=1 \
  --max_input_len=2048 \
  --max_seq_len=2560 \
  --max_multimodal_len=1536 \
  --max_num_tokens=2560 \
  $FP8_FLAG

echo "✓ Stage 2 done: $ENGINE_DIR/llm/"

# ── (Optional) Sequence-length presets ────────────────────────────
# 取消其中一行的註解來覆蓋上面的預設值：
#
#   Small  : --max_input_len=1440 --max_seq_len=2048 --max_multimodal_len=1024 --max_num_tokens=2048
#   Default: --max_input_len=2048 --max_seq_len=2560 --max_multimodal_len=1536 --max_num_tokens=2560  ← active
#   Medium : --max_input_len=2560 --max_seq_len=3072 --max_multimodal_len=2048 --max_num_tokens=3072
#   Large  : --max_input_len=4096 --max_seq_len=4608 --max_multimodal_len=4096 --max_num_tokens=4608

sleep 5

# ══════════════════════════════════════════════════════════════════
#  STAGE 3 — Build vision encoder engine
# ══════════════════════════════════════════════════════════════════
echo ""
echo "================================================="
echo " Stage 3/3: Building vision encoder engine"
echo " model_type=$MULTIMODAL_TYPE"
echo " WARNING: Requires ~20GB RAM. Takes 8-15 min."
echo "================================================="

TARGET_DTYPE="torch.$VISION_DTYPE_PATCH"
if ! grep -qP "from_pretrained.*torch_dtype=${TARGET_DTYPE}|torch_dtype=${TARGET_DTYPE}.*from_pretrained" "$MULTIMODAL_BUILDER"; then
  echo "Patching multimodal_builder.py → torch_dtype=${TARGET_DTYPE} ..."
  cp "$MULTIMODAL_BUILDER" "${MULTIMODAL_BUILDER}.bak"
  sed -i "/from_pretrained/s/torch_dtype=torch\.\(float32\|float16\|bfloat16\),/torch_dtype=${TARGET_DTYPE},/" "$MULTIMODAL_BUILDER"
  echo "✓ Patch applied (backup: ${MULTIMODAL_BUILDER}.bak)"
else
  echo "torch_dtype=${TARGET_DTYPE} already applied, skipping."
fi

python3 "$TRTLLM_ROOT/examples/models/core/multimodal/build_multimodal_engine.py" \
  --model_type  "$MULTIMODAL_TYPE" \
  --model_path  "$MODEL_PATH" \
  --output_dir  "$ENGINE_DIR/vision" \
  --max_batch_size 1

echo "✓ Stage 3 done: $ENGINE_DIR/vision/"

# ── Smoke test ────────────────────────────────────────────────────
echo ""
echo "================================================="
echo " Smoke test: verifying pipeline end-to-end"
echo "================================================="

mkdir -p /workspace/test_assets

python3 - <<'EOF'
from datasets import load_dataset

ds = load_dataset("HuggingFaceM4/VQAv2", split="validation", streaming=True)
sample = next(iter(ds))

sample["image"].save("/workspace/test_assets/sample.jpg")
print("Q:", sample["question"])
print("A:", sample["answers"])
EOF

python3 "$TRTLLM_ROOT/examples/models/core/multimodal/run.py" \
  --hf_model_dir "$MODEL_PATH" \
  --engine_dir   "$ENGINE_DIR" \
  --input_text   "Describe what you see in this image in one sentence." \
  --image_path   /workspace/test_assets/sample.jpg

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "================================================="
echo " Build complete [$MODEL / $QUANT]. Engine sizes:"
echo "================================================="
du -sh "$ENGINE_DIR/llm/rank0.engine"
du -sh "$ENGINE_DIR/vision/model.engine"
echo ""
