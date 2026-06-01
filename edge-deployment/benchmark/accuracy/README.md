# Accuracy Benchmark — Swift-VLM-Flow

Accuracy evaluation for Qwen2-VL-2B-Instruct on **VQAv2**, **POPE**, and **MME**,
for both the HuggingFace baseline and the TensorRT-LLM engine.

---

## Quick Start

```bash
# Full run: HF baseline + TRT bf16, all three benchmarks
bash run_accuracy_all.sh

# Smoke-test (vqa=20 / pope=30 / mme=50)
bash run_accuracy_all.sh --quick

# Baseline only
bash run_accuracy_all.sh --skip-trt

# TRT only, specific precision
bash run_accuracy_all.sh --skip-baseline \
    --engine-dir /workspace/trt_engines/qwen2vl_2b_int4 \
    --precision int4
```

Results are saved to `/workspace/results/accuracy/`.
A side-by-side comparison table is printed when the script finishes.

---

## Run Individually

### HF Baseline

```bash
python3 run_accuracy_baseline.py
python3 run_accuracy_baseline.py --tasks vqa pope
python3 run_accuracy_baseline.py --tasks vqa --vqa_samples 20
```

Saves to: `results/accuracy/baseline/bf16_<run_id>[_<tag>].json`

### TRT Engine

```bash
python3 run_accuracy_trt.py
python3 run_accuracy_trt.py --tasks vqa --vqa_samples 20
python3 run_accuracy_trt.py \
    --engine_dir /workspace/trt_engines/qwen2vl_2b_int4 \
    --precision int4
```

Saves to: `results/accuracy/trt/<precision>_<run_id>[_<tag>].json`

---

## `run_accuracy_all.sh` — Flag Reference

| Flag | Default | Description |
|---|---|---|
| `--skip-baseline` | false | Skip HF run |
| `--skip-trt` | false | Skip TRT run |
| `--precision PREC` | `bf16` | TRT engine precision label |
| `--engine-dir PATH` | `/workspace/trt_engines/qwen2vl_2b_bf16` | TRT engine directory |
| `--tasks "vqa pope mme"` | all three | Space-separated task list (must be quoted) |
| `--vqa-samples N` | 500 | VQAv2 sample count |
| `--pope-samples N` | 500 | POPE samples per split |
| `--mme-samples N` | all | MME sample count (omit for full dataset) |
| `--quick` | false | Sets vqa=20, pope=30, mme=50 |
| `--output-tag TAG` | — | Append tag to output filenames |

---

## Benchmarks & Scoring

### VQAv2

- **Dataset**: `lmms-lab/VQAv2`, validation split, **500 samples** (seed=42)
- **Prompt**: `"{question} Answer the question using a single word or phrase."`
- **Scoring**: VQAv2 official soft accuracy — `min(# matching annotations / 3, 1.0)` per sample, averaged
- **Output key**: `results.vqa.scores.accuracy` (0–100%)

### POPE

- **Dataset**: `lmms-lab/POPE`, test split, **500 samples × 3 splits**
- **Splits**: `random`, `popular`, `adversarial`
- **Prompt**: `"{question} Please answer yes or no."`
- **Scoring**: Binary Yes/No classification; per-split Accuracy + F1; macro-averaged across splits
- **Output keys**: `results.pope.scores.avg_accuracy`, `results.pope.scores.avg_f1`, `results.pope.scores.per_split`

### MME

- **Dataset**: `lmms-lab/MME`, test split, **full dataset** (~2.8K samples)
- **Tasks**: 10 perception + 4 cognition tasks
- **Prompt**: `"{question} Please answer yes or no."`
- **Scoring**: Pair scoring — 2 pts if both questions correct, 1 pt if one, 0 if neither
- **Output keys**: `results.mme.scores.total_score`, `results.mme.scores.perception_score`, `results.mme.scores.cognition_score`, `results.mme.scores.per_task`

---

## TRT-Specific: Image Resizing

TRT engines have a fixed `--max_multimodal_len` (visual token budget).
`data_loader_acc.resize_img()` clamps each image to **≤ 1296 visual tokens** before TRT inference.
The HF baseline has no such constraint, so large images may be processed at slightly different resolutions — this can introduce a small accuracy gap independent of quantization.

---

## Output JSON Schema

```json
{
  "run_id":    "20260527_012345",
  "backend":   "huggingface-bf16",
  "config": {
    "model": "Qwen2-VL-2B-Instruct",
    "vqa_samples": 500, "pope_samples": 500, "mme_samples": null, "max_new_tokens": 50
  },
  "results": {
    "vqa":  { "scores": { "accuracy": 62.34 }, "per_sample": [...] },
    "pope": { "scores": { "avg_accuracy": 85.12, "avg_f1": 84.90, "per_split": {...} } },
    "mme":  { "scores": { "total_score": 1245.0, "perception_score": 1050.0,
                          "cognition_score": 195.0, "per_task": {...} } }
  }
}
```

---

## File Listing

```
accuracy/
├── config_accuracy.py        # paths, dataset IDs, sample counts, prompt templates
├── data_loader_acc.py        # VQAv2 / POPE / MME loaders; resize_img() for TRT
├── accuracy_metrics.py       # score_vqa, score_pope, score_mme
├── run_accuracy_baseline.py  # HF Transformers runner
├── run_accuracy_trt.py       # TensorRT-LLM runner
├── run_accuracy_all.sh       # orchestrator: runs both, prints comparison table
└── README.md
```

---

*Swift-VLM-Flow — Edge Deployment — CSE 599S, University of Washington*
