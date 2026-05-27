# Benchmark — Swift-VLM-Flow

Evaluation pipeline for Qwen2-VL-2B-Instruct — both **efficiency** (latency, VRAM) and
**accuracy** (VQAv2, POPE, MME) — covering the HuggingFace baseline and the TensorRT-LLM engine.

Part of the **Swift-VLM-Flow** project (CSE 599S, UW).

---

## Pipeline Overview

```
efficiency/
  run_benchmarks.sh       →  Tool 1 + 1b: latency & VRAM
      ├── run_benchmark.py          (PyTorch baseline)
      └── run_benchmark_trt.py      (TensorRT-LLM)
                                        ↓
                              results/efficiency/baseline/<id>.json
                              results/efficiency/trt/<precision>_<id>.json

accuracy/
  run_accuracy_all.sh     →  Tool 2: VQAv2 / POPE / MME scoring
      ├── run_accuracy_baseline.py  (HF Transformers)
      └── run_accuracy_trt.py       (TensorRT-LLM)
                                        ↓
                              results/accuracy/baseline/<id>.json
                              results/accuracy/trt/<precision>_<id>.json

report.py                 →  Tool 3: charts + Markdown report
                                        ↓
                              results/reports/report_<id>.md  +  *.png
```

---

## Subdirectories

| Directory | What it measures | Entry point |
|---|---|---|
| [efficiency/](efficiency/README.md) | Latency (TTFT, decode ms/tok), VRAM — PyTorch and TRT | `efficiency/run_benchmarks.sh` |
| [accuracy/](accuracy/README.md) | VQAv2 accuracy, POPE hallucination, MME perception/cognition — PyTorch and TRT | `accuracy/run_accuracy_all.sh` |

---

## Quick Start

### Efficiency

```bash
cd benchmark/efficiency
bash run_benchmarks.sh --output_tag official_v1

# PyTorch only
bash run_benchmarks.sh --skip_trt --output_tag baseline

# TRT only
bash run_benchmarks.sh --skip_baseline --precision bf16 --output_tag trt_bf16
```

→ See [efficiency/README.md](efficiency/README.md) for the full flag reference.

### Accuracy

```bash
cd benchmark/accuracy
bash run_accuracy_all.sh                        # all tasks, both backends
bash run_accuracy_all.sh --quick                # smoke-test (20 VQA / 30 POPE / 50 MME)
bash run_accuracy_all.sh --skip-trt             # HF baseline only
```

→ See [accuracy/README.md](accuracy/README.md) for scoring details and the full flag reference.

### Report

```bash
# Efficiency only
python3 report.py --efficiency results/efficiency/baseline/<id>.json

# Full comparison (baseline + TRT + accuracy)
python3 report.py \
    --efficiency results/efficiency/baseline/<id>.json \
    --trt        results/efficiency/trt/<precision>_<id>.json \
    --lmms       results/accuracy/baseline/<id>.json \
    --output_tag official_v1
```

---

## Results Directory Layout

```
results/
├── efficiency/
│   ├── baseline/       # JSON from run_benchmark.py
│   └── trt/            # JSON from run_benchmark_trt.py
├── accuracy/
│   ├── baseline/       # JSON from run_accuracy_baseline.py
│   └── trt/            # JSON from run_accuracy_trt.py
└── reports/            # .md reports and .png charts from report.py
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

*Swift-VLM-Flow — Edge Deployment (Kevin) — CSE 599S, University of Washington*