"""
config_accuracy.py
Configuration for the accuracy benchmark (VQAv2, POPE, MME).
Intentionally separate from config.py (efficiency benchmark).
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH   = "/workspace/models/Qwen2-VL-2B-Instruct"
ENGINE_PATH  = "/workspace/trt_engines/qwen2vl_v2"
RESULTS_DIR  = "/workspace/results"
ACC_DIR      = os.path.join(RESULTS_DIR, "accuracy")   # accuracy results go here

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen2-VL-2B-Instruct"

# ── Benchmark Settings ─────────────────────────────────────────────────────────
NUM_WARMUP      = 3
MAX_NEW_TOKENS  = 50    # VQAv2 / MME: short answers; POPE: Yes/No

# ── VQAv2 ──────────────────────────────────────────────────────────────────────
VQAV2_DATASET   = "lmms-lab/VQAv2"
VQAV2_SPLIT     = "validation"
VQAV2_SEED      = 42        # same seed as efficiency benchmark → same 500 samples
VQAV2_SAMPLES   = 500

# ── POPE ───────────────────────────────────────────────────────────────────────
# Three adversarial splits: random / popular / adversarial
# Each split has 3,000 Yes/No questions about object hallucination.
POPE_DATASET    = "lmms-lab/POPE"
POPE_SPLITS     = ["random", "popular", "adversarial"]   # run all three
POPE_SAMPLES    = 300   # per split; set to None for the full split

# ── MME ────────────────────────────────────────────────────────────────────────
# Two sub-categories: perception (14 tasks) + cognition (4 tasks)
MME_DATASET     = "lmms-lab/MME"
MME_SPLIT       = "test"
MME_SAMPLES     = None  # None = run all (~2.8K); MME is small enough

# config_accuracy.py
TRT_MAX_VISUAL_TOKENS = 1296  # = max_hw_dims / 4 = 5184 / 4

# ── Prompt Templates ───────────────────────────────────────────────────────────
# VQAv2: same template as efficiency benchmark for direct comparability
VQAV2_PROMPT    = "{question} Answer the question using a single word or phrase."

# POPE: binary question → force Yes/No
POPE_PROMPT     = "{question} Please answer yes or no."

# MME: perception / cognition tasks also expect Yes/No
MME_PROMPT      = "{question} Please answer yes or no."
