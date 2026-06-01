# Benchmark — Swift-VLM-Flow

Evaluation pipeline for Qwen2-VL-2B-Instruct — **efficiency** (latency, VRAM) and
**accuracy** (VQAv2, POPE, MME) — for the HuggingFace baseline and TensorRT-LLM engine.

---

## Pipeline Overview

```
efficiency/
  run_efficiency_all.sh   →  latency & VRAM (baseline + all TRT precisions)
      ├── run_benchmark_baseline.py  (PyTorch bf16)
      └── run_benchmark_trt.py       (TRT: bf16/int8/int4/smoothquant/fp8/int4_awq/nvfp4)
                                         ↓
                               results/efficiency/baseline/<id>.json
                               results/efficiency/trt/<precision>_<id>.json

accuracy/
  run_accuracy_all.sh     →  VQAv2 / POPE / MME scoring
      ├── run_accuracy_baseline.py  (HF Transformers)
      └── run_accuracy_trt.py       (TensorRT-LLM)
                                         ↓
                               results/accuracy/baseline/<id>.json
                               results/accuracy/trt/<precision>_<id>.json

report.py                 →  charts + Markdown report
                                         ↓
                               results/reports/report_<timestamp>/report.md + img/*.png
```

---

## Quick Start

### Efficiency

```bash
cd /workspace/benchmark/efficiency
bash run_efficiency_all.sh

# Custom sample count / warmup / token budget
bash run_efficiency_all.sh --num_samples 28 --warmup 3 --max_new_tokens 256
```

### Accuracy

```bash
cd /workspace/benchmark/accuracy
bash run_accuracy_all.sh                        # all tasks, baseline + TRT bf16
bash run_accuracy_all.sh --quick                # smoke-test (vqa=20 / pope=30 / mme=50)
bash run_accuracy_all.sh --skip-trt             # HF baseline only
```

### Report

```bash
cd /workspace/benchmark

# Auto-discover the latest JSON per tier (recommended)
python3 report.py --latest

# Use hardcoded INPUT_JSONS list in the script
python3 report.py

# Explicit files
python3 report.py path/to/eff.json path/to/acc.json
```

---

## Results Directory Layout

```
results/
├── efficiency/
│   ├── baseline/   # JSON from run_benchmark_baseline.py
│   └── trt/        # JSON from run_benchmark_trt.py
├── accuracy/
│   ├── baseline/   # JSON from run_accuracy_baseline.py
│   └── trt/        # JSON from run_accuracy_trt.py
└── reports/        # .md reports and .png charts from report.py
```

---

## Metrics

### Efficiency

| Metric | Description |
|---|---|
| `ttft_ms` | Time to First Token (ms) — prefill cost |
| `decode_latency_ms_per_tok` | Decode time per output token (ms/tok) |
| `static_vram_gb` | GPU memory after model/engine load |
| `dynamic_vram_gb` | Per-inference VRAM increment above static baseline |

### Accuracy

| Benchmark | Metric | Description |
|---|---|---|
| VQAv2 | `accuracy` (0–100%) | Soft-match against 10 human annotations |
| POPE | `avg_accuracy`, `avg_f1` | Object hallucination across random/popular/adversarial splits |
| MME | `total_score` | Perception (10 tasks) + Cognition (4 tasks) pair scoring |

---

*Swift-VLM-Flow — Edge Deployment — CSE 599S, University of Washington*
