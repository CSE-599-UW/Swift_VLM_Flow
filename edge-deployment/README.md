# Swift-VLM-Flow — Edge Deployment

**CSE 599S, University of Washington**

## Edge Deployment

The work is split into two phases:

| Phase | Description | Note |
|-------|--------|-------------|
| Phase 1 — PyTorch Baseline | Measure fp16 PyTorch inference on VQAv2; establish reference numbers | What left: `Quality benchmark`
| Phase 2 — TensorRT Engine | Convert to TRT fp16 / fp8, re-run benchmark, measure speedup |

Model under test: **Qwen2-VL-2B-Instruct** (fp16, single-batch, VQAv2 validation set)

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA RTX 5060 Ti (16 GB VRAM) or equivalent ≥ 16 GB VRAM |
| CUDA Driver | ≥ 12.8 (driver 595.x or newer for RTX 50-series) |
| Docker | 24.x or newer with `nvidia-container-toolkit` |
| Disk | ~25 GB free (Docker image ~20 GB + model weights ~5 GB) |
| RAM | ≥ 16 GB system RAM recommended |

> The model requires ~4.55 GB VRAM at load time; peak inference VRAM is ~4.26 GB
> on top of the static footprint. A 16 GB card is comfortable for single-batch runs.

---

## Reproducing the Baseline Benchmark

### Step 1 — Pull the base image (first time only, ~20 GB)

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:0.21.0
```

### Step 2 — Build the custom `vlm-bench` image

```bash
cd edge-deployment/
docker build -t vlm-bench:latest .
```

The `Dockerfile` extends the TensorRT-LLM 0.21.0 image with three additional
packages (`qwen-vl-utils`, `datasets`, `matplotlib`).

### Step 3 — Download model weights (first time only)

```bash
# Outside the container — downloads into your HuggingFace cache
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct
```

Weights are cached at `~/.cache/huggingface/` by default. The container mounts
this cache read-only so no re-download is needed on subsequent runs.

### Step 4 — Start the container

```bash
docker run -it --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$(pwd)":/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vlm-bench:latest \
  bash
```

### Step 5 — Run the benchmark

Inside the container:

```bash
cd /workspace/benchmark
python3 run_benchmark.py
```

Default settings: 50 VQAv2 samples, 3 warmup runs, max 50 new tokens, seed 42.

Optional flags:

```bash
python3 run_benchmark.py \
  --num_samples 50 \
  --warmup 3 \
  --max_new_tokens 50 \
  --output_tag v1
```

### Step 6 — Find your results

After the run completes (≈ 5 minutes), outputs appear under:

```
benchmark/results/
├── raw/
│   └── baseline_YYYYMMDD_HHMMSS.json      # full per-sample data
└── reports/
    ├── baseline_YYYYMMDD_HHMMSS.md        # markdown report
    ├── latency_distribution_*.png
    ├── throughput_vram_*.png
    └── per_sample_latency_*.png
```

---

## Official Baseline Results

Run ID: `20260509_030136` — GPU: NVIDIA GeForce RTX 5060 Ti (16 GB), CUDA 12.8

| Metric | Mean | Std | Median | p95 |
|--------|------|-----|--------|-----|
| TTFT (ms) | 127.8 | 29.0 | 122.5 | 204.9 |
| Total Latency (ms) | 570.4 | 109.4 | 539.6 | 790.6 |
| Throughput (tok/s) | 51.0 | 7.3 | 50.7 | 63.7 |
| Peak VRAM (GB) | 4.26 | 0.02 | 4.25 | 4.32 |
| Model VRAM (GB) | 4.55 | — | — | — |

Full report: [results/baseline_20260509_030136.md](results/baseline_20260509_030136.md)

### TRT Comparison Table (to be filled)

| Metric | Baseline fp16 (PyTorch) | TRT fp16 | TRT fp8 |
|--------|------------------------|----------|---------|
| TTFT (ms) | 127.8 | — | — |
| Total Latency (ms) | 570.4 | — | — |
| Throughput (tok/s) | 51.0 | — | — |
| Peak VRAM (GB) | 4.26 | — | — |
| Model VRAM (GB) | 4.55 | — | — |

---

## Repository Structure

```
edge-deployment/
├── Dockerfile                  # Extends TensorRT-LLM 0.21.0 with benchmark deps
├── README.md                   # This file
├── .gitignore
├── benchmark/
│   ├── config.py               # All paths, hyperparameters, seeds
│   ├── data_loader.py          # VQAv2 loading and prompt formatting
│   ├── metrics.py              # Per-sample measurement and aggregate stats
│   ├── run_benchmark.py        # Main entry point
│   ├── report_generator.py     # Markdown report + matplotlib charts
│   └── README.md               # Benchmark-specific docs
└── results/
    └── baseline_20260509_030136.md   # Official baseline report (Phase 1)
```

> `models/` and `results/raw/*.json` are excluded from git (see `.gitignore`).

---

## Next Steps

**Phase 2 — TensorRT Engine Conversion**

1. Convert Qwen2-VL-2B-Instruct checkpoint to TRT-LLM format using
   `convert_checkpoint.py` from the TensorRT-LLM examples.
2. Build the `.engine` file with `trtllm-build` (fp16, then fp8).
3. Run inference on the **same VQAv2 sample IDs** as the baseline
   (extract `question_id` list from `baseline_20260509_030136_official_v1.json`).
4. Fill the TRT columns in the comparison table above.
5. Target: ≥ 2× throughput improvement over the fp16 PyTorch baseline.

---

*Swift-VLM-Flow — CSE 599S, University of Washington*
