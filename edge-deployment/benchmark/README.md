# VLM Baseline Benchmark

Baseline benchmarking suite for **Qwen2-VL-2B-Instruct** (fp16, PyTorch backend)  
on the VQAv2 dataset. Part of the **Swift-VLM-Flow** project (CSE 599S, UW).

This benchmark establishes the reference performance numbers that TensorRT-optimized
engines will be compared against.

---

## Directory Structure

```
benchmark/
├── config.py              # All paths, hyperparameters, and settings
├── data_loader.py         # VQAv2 dataset loading and prompt formatting
├── metrics.py             # Per-sample measurement and aggregate statistics
├── run_benchmark.py       # Main entry point
├── report_generator.py    # Markdown report + matplotlib chart generation
├── results/
│   ├── raw/               # Raw per-sample JSON results (one file per run)
│   └── reports/           # Markdown reports + PNG charts
└── README.md              # This file
```

---

## Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| **TTFT** | Time to First Token — prefill latency including visual encoding | ms |
| **Total Latency** | Full generation time for up to `MAX_NEW_TOKENS` tokens | ms |
| **Throughput** | Output tokens generated per second | tok/s |
| **Peak VRAM** | Maximum GPU memory allocated during inference | GB |
| **Model VRAM** | Static GPU memory occupied by model weights | GB |

All metrics are reported as: **mean ± std**, **median**, **p95**, **min**, **max**.

---

<!-- ## Quick Start

### 1. Start the Docker container

```bash
docker run -it --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ~/Workspace/Course/CSE599s/Project:/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:0.21.0 \
  bash
```

### 2. Install dependencies

```bash
pip install -r /app/tensorrt_llm/examples/models/core/multimodal/requirements-qwen2vl.txt
pip install datasets matplotlib
```

### 3. Run the benchmark

```bash
cd /workspace/benchmark
python3 run_benchmark.py
```

### 4. Optional arguments

```bash
python3 run_benchmark.py \
  --num_samples 50 \      # Number of VQAv2 samples (default: 50)
  --warmup 3 \            # Warmup runs (default: 3)
  --max_new_tokens 50 \   # Max tokens to generate (default: 50)
  --output_tag v1         # Optional tag for output filenames
``` -->

## Quick Start

### 1. Build the custom Docker image (first time only)

```bash
cd ~/Workspace/Course/CSE599s/Project
docker build -t vlm-bench:latest .
```

### 2. Start the container

```bash
docker run -it --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ~/Workspace/Course/CSE599s/Project:/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vlm-bench:latest \
  bash
```

### 3. Run the benchmark

```bash
cd /workspace/benchmark
python3 run_benchmark.py
```

### 4. Optional arguments

```bash
python3 run_benchmark.py \
  --num_samples 50 \
  --warmup 3 \
  --output_tag v1
```

---

## Output Files

After a run, the following files are created:

```
results/
├── raw/
│   └── baseline_YYYYMMDD_HHMMSS.json     # Full per-sample data + summary
└── reports/
    ├── baseline_YYYYMMDD_HHMMSS.md        # Markdown report
    ├── latency_distribution_*.png         # TTFT + Total Latency histograms
    ├── throughput_vram_*.png              # Throughput + VRAM box plots
    └── per_sample_latency_*.png           # Latency over time line chart
```

---

## Measurement Methodology

- **TTFT** is isolated by running a separate `max_new_tokens=1` generation call.
- **Total Latency** covers a full generation up to `MAX_NEW_TOKENS`.
- All timing uses `torch.cuda.synchronize()` + `time.perf_counter()` to ensure
  GPU completion before timestamps are recorded.
- VRAM stats are reset via `torch.cuda.reset_peak_memory_stats()` before each sample.
- Warmup runs are performed to eliminate JIT and caching overhead.
- VQAv2 subset is sampled with a fixed `random.seed(42)` for reproducibility.

---

## Comparing Against TRT Results

The raw JSON files store all configuration and per-sample data needed for fair
comparison. When TensorRT engines are ready, run inference using the **same
`question_id` list** extracted from the baseline JSON to ensure identical inputs.

The comparison table template is included at the bottom of each generated report:

| Metric | Baseline (fp16 PyTorch) | TRT fp16 | TRT fp8 |
|--------|------------------------|----------|---------|
| TTFT (ms) | ... | — | — |
| Total Latency (ms) | ... | — | — |
| Throughput (tok/s) | ... | — | — |
| Peak VRAM (GB) | ... | — | — |

---

## Configuration

All settings are centralized in `config.py`. Key parameters:

```python
NUM_SAMPLES    = 50       # VQAv2 samples per run
NUM_WARMUP     = 3        # Warmup iterations
MAX_NEW_TOKENS = 50       # Generation budget
VQAV2_SEED     = 42       # Reproducibility seed
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.8.0 (in container) | Inference backend |
| transformers | 4.51.3 (in container) | Model loading |
| qwen-vl-utils | 0.0.8 | Qwen2-VL image preprocessing |
| datasets | latest | VQAv2 loading |
| matplotlib | latest | Chart generation |

---

*Swift-VLM-Flow — Edge Deployment (Kevin) — CSE 599S, University of Washington*
