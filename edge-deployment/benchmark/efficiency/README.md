# Efficiency Benchmark — Swift-VLM-Flow

Self-contained efficiency benchmark for Qwen2-VL-2B-Instruct.
Measures **latency**, **VRAM**, and **throughput** for both the HuggingFace PyTorch
baseline and the TensorRT-LLM engine, on the same VQAv2 samples.

Part of the **Swift-VLM-Flow** project (CSE 599S, UW).

---

## Quick Start

```bash
# Run both baseline + TRT in sequence (recommended)
bash run_benchmarks.sh

# Baseline only (no TRT engine)
bash run_benchmarks.sh --skip_trt

# TRT only (baseline already done)
bash run_benchmarks.sh --skip_baseline --engine_dir /workspace/trt_engines/qwen2vl_v2

# Custom sample count and output tag
bash run_benchmarks.sh --num_samples 100 --output_tag v2_run
```

Results are written to `/workspace/results/efficiency/baseline/` and `/workspace/results/efficiency/trt/`.

---

## Run Individually

### PyTorch Baseline

```bash
python3 run_benchmark.py
python3 run_benchmark.py --num_samples 50 --warmup 3 --max_new_tokens 50
python3 run_benchmark.py --output_tag bf16_v1
```

Loads Qwen2-VL-2B-Instruct in **bf16** via HuggingFace Transformers.
Saves to: `results/efficiency/baseline/<run_id>[_<tag>].json`

### TensorRT-LLM

```bash
python3 run_benchmark_trt.py
python3 run_benchmark_trt.py \
    --engine_dir /workspace/trt_engines/qwen2vl_v2 \
    --precision bf16 \
    --output_tag bf16_v1
```

Requires a pre-built TRT engine with `llm/` and `vision/` subdirectories.
Saves to: `results/efficiency/trt/<run_id>[_<tag>].json`

---

## run_benchmarks.sh — Flag Reference

| Flag | Default | Description |
|---|---|---|
| `--num_samples N` | 50 | VQAv2 samples to evaluate |
| `--warmup W` | 3 | Warmup inference runs (discarded) |
| `--max_new_tokens T` | 50 | Max tokens generated per sample |
| `--output_tag TAG` | — | Append a tag string to output filenames |
| `--engine_dir DIR` | `/workspace/trt_engines/qwen2vl_v2` | TRT engine directory |
| `--precision PREC` | `bf16` | TRT precision label (`bf16`, `fp16`, `fp8`) |
| `--skip_baseline` | — | Skip PyTorch baseline run |
| `--skip_trt` | — | Skip TensorRT run |

---

## What Each Runner Does (Step by Step)

```
1. Load model          →  measure static_vram_gb  (weights in VRAM)
2. Load VQAv2 samples  →  stream N samples from HuggingFace (seed=42, reproducible)
3. Warmup              →  run W inferences, discard results (warms GPU caches)
4. Benchmark loop      →  for each sample:
     a. reset_peak_memory_stats()
     b. generate(max_new_tokens=1)   →  measure TTFT
     c. generate(max_new_tokens=50)  →  measure total_latency_ms
     d. read max_memory_allocated()  →  compute dynamic_vram_gb
5. Aggregate           →  mean / std / median / p95 across all samples
6. Save JSON           →  results/efficiency/{baseline,trt}/<run_id>.json
```

> **Why two generate() calls?**
> TTFT (prefill cost) is isolated by running `max_new_tokens=1` first and recording
> when that token appears. The full decode run (`max_new_tokens=50`) then gives total
> latency. Since both calls use identical inputs their prefill times are equivalent,
> so `decode_latency = total_latency − TTFT` correctly isolates the decode phase.

---

## Metrics Explained

| Metric | Unit | Description |
|---|---|---|
| `ttft_ms` | ms | Time to First Token — prefill (vision encoder + LLM prompt processing) |
| `total_latency_ms` | ms | Wall-clock time from input to last generated token |
| `decode_latency_ms_per_tok` | ms/tok | `(total_latency_ms − ttft_ms) / output_tokens` |
| `static_vram_gb` | GB | VRAM after model weights load, before any inference |
| `dynamic_vram_gb` | GB | Per-inference peak VRAM above static baseline (KV-cache + activations) |
| `output_tokens` | tokens | Number of tokens generated per sample |

Aggregate statistics per metric: **mean, std, median, p95, min, max**.

---

## Output JSON Schema

```json
{
  "run_id":         "20260527_012345",
  "timestamp":      "2026-05-27T01:23:45",
  "config": {
    "model":          "Qwen2-VL-2B-Instruct",
    "precision":      "bf16",
    "backend":        "pytorch",
    "num_samples":    50,
    "num_warmup":     3,
    "max_new_tokens": 50,
    "dataset":        "lmms-lab/VQAv2",
    "split":          "validation",
    "seed":           42
  },
  "static_vram_gb": 4.123,
  "summary": {
    "ttft_ms":                   { "mean": 312.5, "std": 18.2, "median": 308.1, "p95": 350.0, "min": 290.0, "max": 380.0 },
    "decode_latency_ms_per_tok": { "mean": 22.4,  "std": 1.1,  "median": 22.1,  "p95": 24.5,  "min": 20.1,  "max": 26.3 },
    "dynamic_vram_gb":           { "mean": 0.51,  "std": 0.02, "median": 0.51,  "p95": 0.55,  "min": 0.49,  "max": 0.57 },
    "output_tokens":             { "mean": 12.3,  "std": 3.1,  "median": 11.0,  "p95": 18.0,  "min": 5.0,   "max": 22.0 }
  },
  "per_sample": [
    {
      "question_id": 123456,
      "question": "What color is the car?",
      "predicted_answer": "The car is red.",
      "ground_truth_answers": ["red", "red", "red"],
      "ttft_ms": 310.2,
      "total_latency_ms": 598.4,
      "decode_latency_ms_per_tok": 21.8,
      "dynamic_vram_gb": 0.51,
      "output_tokens": 13
    }
  ]
}
```

Both `run_benchmark.py` (baseline) and `run_benchmark_trt.py` (TRT) write this
same schema so results can be fed directly to `../report.py` for comparison charts.

---

## File Listing

```
efficiency/
├── config.py             # paths, model settings, benchmark hyperparameters
├── data_loader.py        # VQAv2 streaming loader (HuggingFace, seed=42)
├── metrics.py            # LatencyTimer, VRAM measurement, aggregate statistics
├── run_benchmark.py      # PyTorch baseline runner
├── run_benchmark_trt.py  # TensorRT-LLM runner
├── run_benchmarks.sh     # orchestrator: runs both and reports exit codes
└── README.md
```

---

## Results Directory

```
results/efficiency/
├── baseline/    # JSON outputs from run_benchmark.py
└── trt/         # JSON outputs from run_benchmark_trt.py
```

Pass these JSONs to `../report.py` to generate comparison charts and a Markdown report:

```bash
python3 ../report.py \
    --efficiency results/efficiency/baseline/<id>.json \
    --trt        results/efficiency/trt/<id>.json \
    --output_tag official_v1
```

---

*Swift-VLM-Flow — Edge Deployment (Kevin) — CSE 599S, University of Washington*