"""config_spec.py — configuration for the Qwen2.5-VL-7B EAGLE3 speculative-decoding benchmark.

Runtime: NVIDIA TensorRT-Edge-LLM (C++ `llm_inference`), driven as a subprocess.
Target box: GB10 (aarch64, sm_121, CUDA 13.0). See env.sh for the shell env the
binaries need (CUDA_HOME, TRT_PACKAGE_DIR, LD_LIBRARY_PATH, EDGE_LLM_PATH).
"""
import os

# ── Models ──────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-VL-7B-Instruct"
DRAFT_MODEL = "Rayzl/qwen2.5-vl-7b-eagle3-sgl"
MODEL_NAME  = "Qwen2.5-VL-7B-Instruct"
PRECISION   = os.environ.get("SD_PRECISION", "bf16")   # bf16 on GB10; "fp8" once B200 ONNX is in

# ── TensorRT-Edge-LLM paths (overridable via env.sh) ─────────────────────────
EDGE_LLM_PATH = os.path.expanduser(os.environ.get("EDGE_LLM_PATH", "~/TensorRT-Edge-LLM"))
WORKSPACE_DIR = os.path.expanduser(os.environ.get("WORKSPACE_DIR", "~/tensorrt-edgellm-workspace"))
ENGINE_LLM_DIR         = os.path.join(WORKSPACE_DIR, MODEL_NAME, "engines", "llm")          # EAGLE base+draft (SD-on)
ENGINE_LLM_VANILLA_DIR = os.path.join(WORKSPACE_DIR, MODEL_NAME, "engines", "llm_vanilla")  # plain base (SD-off baseline)
ENGINE_VISUAL_DIR      = os.path.join(WORKSPACE_DIR, MODEL_NAME, "engines", "visual")
LLM_INFERENCE_BIN      = os.path.join(EDGE_LLM_PATH, "build", "examples", "llm", "llm_inference")

# ── EAGLE3 / spec-decode params (llm_inference flags) ────────────────────────
# Defaults match the SGLang config the Rayzl draft was trained with where sensible.
SPEC_DRAFT_TOPK  = int(os.environ.get("SD_DRAFT_TOPK", "10"))   # --specDraftTopK
SPEC_DRAFT_STEP  = int(os.environ.get("SD_DRAFT_STEP", "6"))    # --specDraftStep
SPEC_VERIFY_TREE = int(os.environ.get("SD_VERIFY_TREE", "60"))  # --specVerifyTreeSize
# Engine build tree sizes (llm_build --maxVerifyTreeSize / --maxDraftTreeSize)
MAX_VERIFY_TREE_SIZE = 60
MAX_DRAFT_TREE_SIZE  = 60

# ── Benchmark params ────────────────────────────────────────────────────────
NUM_SAMPLES    = 60
NUM_WARMUP     = 3
MAX_NEW_TOKENS = 256
SAMPLE_SEED    = 42
DATASET        = "lmms-lab/llava-bench-in-the-wild"

# ── Output ──────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results", "speculative")
)
