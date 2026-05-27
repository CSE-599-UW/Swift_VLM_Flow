"""
config.py
Central configuration for the VLM Baseline Benchmark.
All paths, hyperparameters, and benchmark settings are defined here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "/workspace/models/Qwen2-VL-2B-Instruct"
RESULTS_DIR = "/workspace/results"
RAW_DIR = os.path.join(RESULTS_DIR, "efficiency")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# ── Model Settings ─────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen2-VL-2B-Instruct"
PRECISION = "bf16"
BACKEND = "pytorch"
DEVICE = "cuda"

# ── Benchmark Settings ─────────────────────────────────────────────────────────
NUM_WARMUP = 3           # Number of warmup runs before benchmarking
NUM_SAMPLES = 50         # Number of VQAv2 samples to evaluate
MAX_NEW_TOKENS = 50      # Maximum tokens to generate per sample
# MIN_NEW_TOKENS = 20
BATCH_SIZE = 1           # Batch size (keep 1 for latency measurement)

# ── VQAv2 Dataset Settings ─────────────────────────────────────────────────────
VQAV2_DATASET = "lmms-lab/VQAv2"   # HuggingFace dataset identifier
VQAV2_SPLIT = "validation"          # Dataset split to use
VQAV2_SEED = 42                     # Random seed for reproducible sampling

# ── Prompt Template ────────────────────────────────────────────────────────────
# Short QA format: single-turn, concise answer expected
SYSTEM_PROMPT = None   # Qwen2-VL does not require a system prompt
USER_PROMPT_TEMPLATE = "{question} Answer in a complete sentence."
