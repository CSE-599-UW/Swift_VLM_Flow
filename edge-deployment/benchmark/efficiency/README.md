# Efficiency Benchmark — Swift-VLM-Flow

Measures **latency** (TTFT, decode ms/tok) and **VRAM** for Qwen2-VL-2B-Instruct,
covering the HuggingFace PyTorch baseline and all TensorRT-LLM engine precisions,
on LLaVA-Bench (In-the-Wild) samples.

---

## Quick Start

```bash
# Run baseline + all TRT precisions in sequence
bash run_efficiency_all.sh

# Custom sample count / warmup / token budget
bash run_efficiency_all.sh --num_samples 28 --warmup 3 --max_new_tokens 256

# Skip the PyTorch baseline, run TRT only
bash run_efficiency_all.sh --no_baseline
```

Results are written to:
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
python3 run_benchmark_trt.py \
    --engine_dir /workspace/trt_engines/qwen2vl_2b_fp8 \
    --precision  fp8 \
    --output_tag fp8_v1
```

Requires a pre-built TRT engine at `--engine_dir` (must contain `llm/` and `vision/` subdirs).
VRAM measured via `nvidia-smi` (TRT-LLM allocates outside PyTorch's allocator).
Saves to: `results/efficiency/trt/<precision>_<run_id>[_<tag>].json`

---

## `run_efficiency_all.sh` — Flag Reference

| Flag | Default | Description |
|---|---|---|
| `--num_samples N` | 50 | LLaVA-Bench samples to evaluate |
| `--warmup W` | 3 | Warmup inference runs (discarded) |
| `--max_new_tokens T` | 256 | Max tokens generated per sample |
| `--no_baseline` | — | Skip the PyTorch baseline |

Runs in order: **baseline → bf16 → int8 → int4 → smoothquant → fp8 → int4_awq → nvfp4**.
Any TRT tier is skipped automatically if its engine directory does not exist.

Engine directories expected under `/workspace/trt_engines/`:

| Precision | Engine Dir | Precision Type |
|---|---|---|
| `bf16` | `qwen2vl_2b_bf16` | W16A16 baseline |
| `int8` | `qwen2vl_2b_int8` | W8A16 weight-only |
| `int4` | `qwen2vl_2b_int4` | W4A16 weight-only |
| `smoothquant` | `qwen2vl_2b_smoothquant` | W8A8 |
| `fp8` | `qwen2vl_2b_fp8` | W8A8 FP8 |
| `int4_awq` | `qwen2vl_2b_int4_awq` | W4A16 AWQ |
| `nvfp4` | `qwen2vl_2b_nvfp4` | W4A8 FP4, Blackwell only |

---

## Dataset

Both runners use **LLaVA-Bench (In-the-Wild)** (`lmms-lab/llava-bench-in-the-wild`),
a 60-sample visual-question benchmark with three categories: `conv`, `detail`, `complex`.
Default: all 60 samples (`category='all'`).

---

## Measurement Methodology

```
1. Load model/engine   →  record static_vram_gb
2. Load LLaVA-Bench    →  first N samples, seed=42
3. Warmup              →  W runs, discarded
4. Benchmark loop (per sample):
     a. generate(max_new_tokens=1)   →  TTFT
     b. generate(max_new_tokens=256) →  total_latency_ms
     c. peak VRAM delta              →  dynamic_vram_gb
5. Aggregate           →  mean / std / median / p95 / min / max
6. Save JSON
```

> **Why two `generate()` calls?**
> TTFT isolates prefill cost by running `max_new_tokens=1`.
> The full decode run gives total latency.
> `decode_latency = (total_latency − TTFT) / output_tokens`.

---

## Metrics

| Metric | Unit | Description |
|---|---|---|
| `ttft_ms` | ms | Time to First Token — prefill (vision encoder + LLM prompt processing) |
| `decode_latency_ms_per_tok` | ms/tok | `(total_latency_ms − ttft_ms) / output_tokens` |
| `static_vram_gb` | GB | VRAM after weights load, before inference |
| `dynamic_vram_gb` | GB | Per-inference peak VRAM above static baseline |
| `output_tokens` | tokens | Tokens generated per sample |

---

## Output JSON Schema

```json
{
  "run_id":   "20260527_012345",
  "config": {
    "model": "Qwen2-VL-2B-Instruct", "precision": "bf16", "backend": "pytorch",
    "num_samples": 50, "num_warmup": 3, "max_new_tokens": 256
  },
  "static_vram_gb": 4.123,
  "summary": {
    "ttft_ms":                   { "mean": 312.5, "std": 18.2, "p95": 350.0 },
    "decode_latency_ms_per_tok": { "mean": 22.4,  "std": 1.1,  "p95": 24.5  },
    "dynamic_vram_gb":           { "mean": 0.51,  "std": 0.02, "p95": 0.55  },
    "output_tokens":             { "mean": 12.3,  "std": 3.1,  "p95": 18.0  }
  },
  "per_sample": [...]
}
```

TRT runs add `"engine_dir"` and `"precision"` to `config`.

---

## File Listing

```
efficiency/
├── config.py                 # paths, model settings, benchmark hyperparameters
├── data_loader.py            # LLaVA-Bench loader (seed=42)
├── metrics.py                # LatencyTimer, VRAM measurement, aggregation
├── run_benchmark_baseline.py # PyTorch runner
├── run_benchmark_trt.py      # TensorRT-LLM runner
├── run_efficiency_all.sh     # orchestrator: baseline + all TRT precisions
└── README.md
```

---

*Swift-VLM-Flow — Edge Deployment — CSE 599S, University of Washington*
