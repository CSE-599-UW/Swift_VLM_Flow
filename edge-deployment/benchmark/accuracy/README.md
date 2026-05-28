# Accuracy Benchmark — Swift-VLM-Flow

Custom accuracy evaluation suite for Qwen2-VL-2B-Instruct on three standard benchmarks:
**VQAv2**, **POPE**, and **MME** — for both the HuggingFace baseline and the TensorRT-LLM engine.

Part of the **Swift-VLM-Flow** project (CSE 599S, UW).

---

## Why a separate accuracy suite?

The parent `benchmark/` directory measures *efficiency* (latency, VRAM).
This subdirectory measures *accuracy* — how well predictions match ground truth —
using the same VQAv2 samples (seed=42) as the efficiency run, so accuracy degradation
from TRT compression can be isolated cleanly.

Both the HF baseline runner and the TRT runner produce the **same JSON schema**,
so results diff directly without post-processing.

---

## Quick Start

```bash
# Full run: HF baseline + TRT engine, all three benchmarks
bash run_accuracy_all.sh

# Quick smoke-test (20 VQA / 30 POPE / 50 MME samples)
bash run_accuracy_all.sh --quick

# Baseline only (no TRT engine available)
bash run_accuracy_all.sh --skip-trt

# TRT only, skip baseline if already done
bash run_accuracy_all.sh --skip-baseline --precision bf16

# Custom engine path and precision label
bash run_accuracy_all.sh \
    --engine-dir /workspace/trt_engines/qwen2vl_int4 \
    --precision int4
```

Results are saved to `/workspace/results/accuracy/` and a side-by-side comparison
table is printed to the terminal when the script finishes.

---

## Run Individually

### HF Baseline

```bash
python3 run_accuracy_baseline.py                         # all tasks, default sample counts
python3 run_accuracy_baseline.py --tasks vqa pope        # subset of tasks
python3 run_accuracy_baseline.py --tasks vqa --vqa_samples 20  # quick test
```

Saves to: `results/accuracy/baseline/bf16_<run_id>[_<tag>].json`

### TRT Engine

```bash
python3 run_accuracy_trt.py                              # all tasks
python3 run_accuracy_trt.py --tasks vqa --vqa_samples 20
python3 run_accuracy_trt.py \
    --engine_dir /workspace/trt_engines/qwen2vl_int4 \
    --precision int4
```

Saves to: `results/accuracy/trt/<precision>_<run_id>[_<tag>].json`

---

## run_accuracy_all.sh — Flag Reference

| Flag | Default | Description |
|---|---|---|
| `--skip-baseline` | false | Skip HF run; reuse the most recent baseline JSON |
| `--skip-trt` | false | Skip TRT run; reuse the most recent TRT JSON |
| `--precision PREC` | `bf16` | TRT engine precision label (`bf16`, `fp16`, `fp8`, `int4`, `int8`) |
| `--engine-dir PATH` | `/workspace/trt_engines/qwen2vl` | TRT engine directory |
| `--tasks "vqa pope mme"` | all three | Space-separated task list (must be quoted) |
| `--vqa-samples N` | 500 | VQAv2 sample count |
| `--pope-samples N` | 300 | POPE samples per split |
| `--mme-samples N` | all | MME sample count |
| `--quick` | false | Sets vqa=20, pope=30, mme=50 for a fast smoke-test |
| `--output-tag TAG` | — | Append a tag string to output filenames |

---

## Benchmarks & Scoring

### VQAv2

- **Dataset**: `lmms-lab/VQAv2`, validation split, **500 samples** (seed=42 — identical to efficiency run)
- **Prompt**: `"{question} Answer the question using a single word or phrase."`
- **Scoring**: VQAv2 official soft accuracy
  - Each question has 10 human annotations
  - `score = min(# annotators who gave this answer / 3, 1.0)`
  - Final accuracy = mean score × 100%
- **Output key**: `results.vqa.scores.accuracy` (0–100%)

### POPE

- **Dataset**: `lmms-lab/POPE`, test split, **300 samples × 3 splits**
- **Splits**: `random`, `popular`, `adversarial` (increasing object hallucination difficulty)
- **Prompt**: `"{question} Please answer yes or no."`
- **Scoring**: Binary classification (Yes/No) with confusion matrix
  - Per split: Accuracy, Precision, Recall, F1
  - Reported: `avg_accuracy` and `avg_f1` (macro-average across all 3 splits)
- **Output keys**: `results.pope.scores.avg_accuracy`, `results.pope.scores.avg_f1`, `results.pope.scores.per_split`

### MME

- **Dataset**: `lmms-lab/MME`, test split, **full dataset** (~2.8K samples)
- **Tasks**: 10 perception tasks + 4 cognition tasks
- **Prompt**: `"{question} Please answer yes or no."`
- **Scoring**: Official MME pair scoring
  - Each image has 2 questions (positive + negative)
  - Per image: 2 pts if both correct, 1 pt if one correct, 0 pts if neither
  - `perception_score` = sum across 10 perception tasks
  - `cognition_score` = sum across 4 cognition tasks
  - `total_score` = perception + cognition
- **Output keys**: `results.mme.scores.total_score`, `results.mme.scores.perception_score`, `results.mme.scores.cognition_score`

---

## Output JSON Schema

Both `run_accuracy_baseline.py` and `run_accuracy_trt.py` write the same schema:

```json
{
  "run_id":    "20260527_012345",
  "timestamp": "2026-05-27T01:23:45",
  "backend":   "huggingface-bf16",
  "config": {
    "model":          "Qwen2-VL-2B-Instruct",
    "vqa_samples":    500,
    "pope_samples":   500,
    "mme_samples":    null,
    "max_new_tokens": 50
  },
  "results": {
    "vqa":  { "scores": { "accuracy": 62.34, "num_samples": 500, ... }, "per_sample": [...] },
    "pope": { "scores": { "avg_accuracy": 85.12, "avg_f1": 84.90, "per_split": {...} }, "per_split_samples": {...} },
    "mme":  { "scores": { "total_score": 1245.0, "perception_score": 1050.0, "cognition_score": 195.0, "per_task": {...} }, "per_sample": [...] }
  }
}
```

---

## File Listing

```
accuracy/
├── config_accuracy.py        # all paths, dataset IDs, sample counts, prompt templates
├── data_loader_acc.py        # VQAv2 / POPE / MME loaders; resize_img() for TRT token budget
├── accuracy_metrics.py       # scoring functions: score_vqa, score_pope, score_mme
├── run_accuracy_baseline.py  # HF Transformers runner (bf16, greedy decoding)
├── run_accuracy_trt.py       # TensorRT-LLM runner (same tasks, same JSON schema)
├── run_accuracy_all.sh       # orchestrator: runs both, prints side-by-side comparison
└── README.md
```

---

## TRT-Specific: Image Resizing

TRT engines are built with a fixed `--max_multimodal_len` (visual token budget).
Before TRT inference, `data_loader_acc.resize_img()` clamps each image to
**≤ 1296 visual tokens** (the tighter of `max_hw_dims` and `max_multimodal_len` constraints).

The HF baseline never calls this function — it has no token-count constraint.
This means HF and TRT may process slightly different image resolutions for large images,
which can introduce a small accuracy gap independent of quantization.

---

## Results Directory

```
results/accuracy/
├── baseline/    # JSON outputs from run_accuracy_baseline.py
└── trt/         # JSON outputs from run_accuracy_trt.py
```

---

*Swift-VLM-Flow — Edge Deployment (Kevin) — CSE 599S, University of Washington*