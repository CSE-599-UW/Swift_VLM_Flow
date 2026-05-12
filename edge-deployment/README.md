# Edge Deployment — Swift-VLM-Flow

**CSE 599S, University of Washington**

Kevin's role in this sub-project is baseline benchmarking and TensorRT engine
conversion. The benchmark uses a two-tool approach — `run_benchmark.py` for
efficiency metrics and `lmms-eval` for accuracy — each tool doing what it is
best at. The goal is to measure performance before and after TRT optimization
to quantify the speedup from TensorRT-LLM fp16 and fp8 engine conversion.

---

## Architecture

```
run_benchmark.py             →  TTFT, Throughput, VRAM       (efficiency)
lmms-eval                    →  VQAv2 exact_match             (accuracy)
generate_combined_report.py  →  merged report + charts
```

---

## Hardware Requirements

- GPU: NVIDIA RTX 5060 Ti or equivalent (16 GB VRAM recommended)
- CUDA Driver: 12.8+
- OS: Ubuntu 22.04

---

## Quick Start

### Step 1 — Clone and get model weights

```bash
git clone https://github.com/CSE-599-UW/Swift_VLM_Flow.git
cd Swift_VLM_Flow/edge-deployment

# Download model — see models/README.md for Google Drive link
# OR via HuggingFace:
pip install huggingface_hub
hf download Qwen/Qwen2-VL-2B-Instruct \
  --local-dir ./models/Qwen2-VL-2B-Instruct
```

### Step 2 — Build Docker image (first time only, ~10-15 min)

```bash
docker build -t vlm-bench:latest .
```

### Step 3 — Start container

```bash
docker run -it --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/Workspace/Course/CSE599s/Project/Swift_VLM_Flow/edge-deployment/models:/workspace/models \
  vlm-bench:latest bash
```
<!-- # -v ~/Workspace/Course/CSE599s/Project/models:/workspace/models \ -->


### Step 4 — Run efficiency benchmark (inside container)

```bash
cd /workspace/benchmark
python3 run_benchmark.py --num_samples 50 --warmup 3
# Output: results/raw/*.json + results/reports/*.md + charts
```

### Step 5 — Run accuracy benchmark (inside container)

```bash
python -m lmms_eval \
  --model qwen2_vl \
  --model_args pretrained=/workspace/models/Qwen2-VL-2B-Instruct \
  --tasks vqav2_val \
  --batch_size 1 \
  --limit 500 \
  --output_path /workspace/results/lmms_eval/
# Output: results/lmms_eval/**/*_results.json
```

### Step 6 — Generate combined report

```bash
python3 generate_combined_report.py
# Output: results/reports/combined_*.md + combined_summary_*.png
```

---

## Why Two Separate Tools?

`run_benchmark.py` and `lmms-eval` serve fundamentally different purposes and
cannot share a single configuration. `lmms-eval` uses `max_new_tokens=16` and
short-answer prompts specifically optimized for accuracy evaluation against
VQAv2 ground-truth annotations — this would distort throughput measurements by
generating far fewer tokens than real inference. `run_benchmark.py` uses
`min_new_tokens=20` for stable throughput measurement with proper warmup and
per-sample VRAM tracking, conditions that are inappropriate for accuracy
evaluation. Keeping the tools separate ensures each metric is measured under
the correct conditions.

---

## Benchmark Metrics

| Metric | Tool | Description |
|--------|------|-------------|
| TTFT (ms) | run_benchmark.py | Time to First Token |
| Total Latency (ms) | run_benchmark.py | Full generation time |
| Throughput (tok/s) | run_benchmark.py | Output tokens per second |
| Peak VRAM (GB) | run_benchmark.py | Max GPU memory during inference |
| VQAv2 Accuracy | lmms-eval | Exact match, standard research metric |

---

## Official Baseline Results

Qwen2-VL-2B-Instruct, fp16, PyTorch — Run ID: `20260509_030136`

| Metric | Value |
|--------|-------|
| TTFT (ms) | 127.8 |
| Total Latency (ms) | 570.4 |
| Throughput (tok/s) | 51.0 |
| Peak VRAM (GB) | 4.26 |
| Model VRAM (GB) | 4.55 |
| VQAv2 Accuracy | 83.4% (500 samples) |

---

## Output Files

```
results/
├── raw/                    # per-sample efficiency JSON (gitignored)
├── lmms_eval/              # lmms-eval accuracy JSON (gitignored)
└── reports/                # markdown reports + PNG charts (tracked in git)
```

---

## TRT Comparison Table

| Metric | PyTorch fp16 | TRT fp16 | TRT fp8 |
|--------|-------------|----------|---------|
| TTFT (ms) | 127.8 | — | — |
| Throughput (tok/s) | 51.0 | — | — |
| Peak VRAM (GB) | 4.26 | — | — |
| VQAv2 Accuracy | 83.4% | — | — |
| Speedup | 1.0× | — | — |

---

## Next Steps

- TRT fp16 engine conversion (`convert_checkpoint.py` → `trtllm-build`)
- TRT fp8 quantization
- Fill TRT columns in comparison table above

---

## Related Papers

- LiteVLM (arXiv:2506.07416) — edge VLM, FP8, 2.5×–3.2× speedup
- MBQ CVPR 2025 — Qwen2-VL quantization with lmms-eval
- GRACE (arXiv:2601.22709) — Qwen2-VL-2B INT4 baseline numbers
- Edge Reliability Gap (arXiv:2603.26769) — VQAv2 on RTX hardware

---

*Swift-VLM-Flow — CSE 599S, University of Washington*
