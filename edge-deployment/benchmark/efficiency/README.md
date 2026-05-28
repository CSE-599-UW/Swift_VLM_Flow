# Efficiency Benchmark — Swift-VLM-Flow

Self-contained efficiency benchmark for Qwen2-VL-2B-Instruct.
Measures **Speed**, **Memory usage** for both the HuggingFace PyTorch
baseline and the TensorRT-LLM engine, on the same LLaVA-Bench (In-the-Wild) samples.


---

## Quick Start

```bash
# Run baseline + all TRT precisions in sequence (recommended)
bash run_efficiency_all.sh

# Custom sample count / warmup / token budget
bash run_efficiency_all.sh --num_samples 28 --warmup 3 --max_new_tokens 256
```

Results are written to 
- `/workspace/results/efficiency/baseline/`
- `/workspace/results/efficiency/trt/`

---

## Run Individually

### PyTorch Baseline

```bash
python3 run_benchmark_baseline.py
python3 run_benchmark_baseline.py --num_samples 50 --warmup 3 --max_new_tokens 256
python3 run_benchmark_baseline.py --output_tag bf16_v1
```

Loads Qwen2-VL-2B-Instruct in **bf16** via HuggingFace Transformers.
Saves to: `results/efficiency/baseline/<run_id>[_<tag>].json`

### TensorRT-LLM

```bash
python3 run_benchmark_trt.py
python3 run_benchmark_trt.py \
    --engine_dir /workspace/trt_engines/qwen2vl_2b_fp8 \
    --precision fp8 \
    --output_tag fp8_v1
```

Requires a pre-built TRT engine at the given `--engine_dir`.
VRAM is measured via `nvidia-smi` (TRT-LLM allocates outside PyTorch's allocator).
Saves to: `results/efficiency/trt/<precision>_<run_id>[_<tag>].json`

---

## `run_efficiency_all.sh` — Flag Reference

| Flag | Default | Description |
|---|---|---|
| `--num_samples N` | 50 | LLaVA-Bench samples to evaluate |
| `--warmup W` | 3 | Warmup inference runs (discarded) |
| `--max_new_tokens T` | 256 | Max tokens generated per sample |

The script runs the PyTorch baseline first, then iterates over all TRT precision
variants in order: **bf16 → fp8 → int8 → int4 → int4_awq**.
Each TRT run is skipped automatically if its engine directory does not exist.

Engine directories expected under `/workspace/trt_engines/`:

| Precision | Engine Dir |
|---|---|
| `bf16` | `qwen2vl_2b_bf16` |
| `fp8` | `qwen2vl_2b_fp8` |
| `int8` | `qwen2vl_2b_int8` |
| `int4` | `qwen2vl_2b_int4` |
| `int4_awq` | `qwen2vl_2b_int4_awq` |

---

## Dataset

Both runners use **LLaVA-Bench (In-the-Wild)** (`lmms-lab/llava-bench-in-the-wild`),
a 60-sample open-ended visual-question benchmark with three categories:

| Category | ~Samples | Output Length | Use |
|---|---|---|---|
| `complex` | 20 | 150–200 tokens | Best for decode latency |
| `detail` | 20 | 100–150 tokens | Medium output |
| `conv` | 20 | 50–100 tokens | Short output |
| `all` | 60 | mixed | Default |

To use a specific category, edit `data_loader.load_llava_bench_samples(category=...)` in the runner scripts.

---

## What Each Runner Does (Step by Step)

```
1. Load model          →  measure static_vram_gb  (weights in VRAM)
2. Load LLaVA-Bench    →  first N samples from the dataset (category='all')
3. Warmup              →  run W inferences, discard results (warms GPU caches)
4. Benchmark loop      →  for each sample:
     a. reset_peak_memory_stats()
     b. generate(max_new_tokens=1)   →  measure TTFT
     c. generate(max_new_tokens=256) →  measure total_latency_ms
     d. read peak memory             →  compute dynamic_vram_gb
5. Aggregate           →  mean / std / median / p95 across all samples
6. Save JSON           →  results/efficiency/{baseline,trt}/<run_id>.json
```

> **Why two generate() calls?**
> TTFT (prefill cost) is isolated by running `max_new_tokens=1` first and recording
> when that token appears. The full decode run (`max_new_tokens=256`) then gives total
> latency. Since both calls use identical inputs their prefill times are equivalent,
> so `decode_latency = total_latency − TTFT` correctly isolates the decode phase.

---

## Metrics Explained

| Metric | Unit | Description |
|---|---|---|
| `ttft_ms` | ms | Time to First Token — prefill (vision encoder + LLM prompt processing) |
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
    "max_new_tokens": 256,
    "dataset":        "lmms-lab/llava-bench-in-the-wild",
    "split":          "train",
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
      "ground_truth_answers": ["The car is red."],
      "ttft_ms": 310.2,
      "total_latency_ms": 598.4,
      "decode_latency_ms_per_tok": 21.8,
      "dynamic_vram_gb": 0.51,
      "output_tokens": 13
    }
  ]
}
```

TRT runs add `"engine_dir"` to the `config` block and prefix the filename with the
precision label: `<precision>_<run_id>[_<tag>].json`.

Both runners write the same schema so results can be fed directly to `../report.py`
for comparison charts.

---

## File Listing

```
efficiency/
├── config.py                 # paths, model settings, benchmark hyperparameters
├── data_loader.py            # LLaVA-Bench loader (HuggingFace, seed=42)
├── metrics.py                # LatencyTimer, VRAM measurement, aggregate statistics
├── run_benchmark_baseline.py # PyTorch baseline runner
├── run_benchmark_trt.py      # TensorRT-LLM runner
├── run_efficiency_all.sh     # orchestrator: baseline + all TRT precisions
└── README.md
```

---

## Results Directory

```
results/efficiency/
├── baseline/    # JSON outputs from run_benchmark_baseline.py
└── trt/         # JSON outputs from run_benchmark_trt.py
```

Pass these JSONs to `../report.py` to generate comparison charts and a Markdown report:

```bash
python3 ../report.py \
    --efficiency results/efficiency/baseline/<id>.json \
    --trt        results/efficiency/trt/<precision>_<id>.json \
    --output_tag official_v1
```

---

*Swift-VLM-Flow — Edge Deployment (Kevin) — CSE 599S, University of Washington*
