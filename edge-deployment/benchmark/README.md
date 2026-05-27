# Benchmark — Swift-VLM-Flow

Three-tool benchmark pipeline for Qwen2-VL-2B-Instruct evaluation.
Part of the **Swift-VLM-Flow** project (CSE 599S, UW).

---

## Pipeline Overview

```
Tool 1  →  Efficiency (PyTorch baseline)    →  results/efficient/baseline/<id>.json
Tool 1b →  Efficiency (TensorRT-LLM)        →  results/efficient/trt/<id>.json
Tool 2  →  Accuracy   (lmms-eval / VQAv2)  →  results/lmms_eval/.../results.json
Tool 3  →  Report     (charts + Markdown)   →  results/reports/report_<id>.md
```

---

## Tool 1 — Efficiency Benchmark (PyTorch baseline)

```bash
python3 run_benchmark.py [--num_samples N] [--warmup W] [--max_new_tokens T] [--output_tag TAG]
```

Default: `--num_samples 50 --warmup 3 --max_new_tokens 50`

Loads Qwen2-VL-2B-Instruct in **bf16** via HuggingFace Transformers. Runs VQAv2 samples,
records per-sample metrics, and writes a JSON to `results/efficient/baseline/`.

**Metrics recorded per sample:**

| Metric | Description |
|---|---|
| `ttft_ms` | Time to First Token (ms) — prefill cost |
| `decode_latency_ms_per_tok` | Decode time per output token (ms/tok) |
| `dynamic_vram_gb` | Per-inference VRAM increment above static baseline (GB) |
| `static_vram_gb` | VRAM after model weights load, before any inference (GB) |
| `output_tokens` | Number of generated tokens |

---

## Tool 1b — Efficiency Benchmark (TensorRT-LLM)

```bash
python3 run_benchmark_trt.py \
    [--num_samples 50] [--warmup 3] [--max_new_tokens 50] \
    [--engine_dir /workspace/trt_engines/qwen2vl] \
    [--precision bf16|fp16|fp8] \
    [--output_tag TAG]
```

Mirrors `run_benchmark.py` exactly — same VQAv2 samples (via seed), same metrics,
same JSON schema — so results are directly comparable. Writes to `results/efficient/trt/`.

Requires a pre-built TRT engine directory with `llm/` and `vision/` subdirectories.

---

## Tool 2 — Accuracy Benchmark (lmms-eval)

```bash
python -m lmms_eval --model qwen2_vl \
  --model_args pretrained=/workspace/models/Qwen2-VL-2B-Instruct \
  --tasks vqav2_val --batch_size 1 --limit 500 \
  --output_path /workspace/results/lmms_eval/
```

Measures VQAv2 `exact_match` accuracy using the standard lmms-eval toolkit
(used by MBQ, LiteVLM, GRACE for apples-to-apples comparison).

---

## Tool 3 — Report Generator

```bash
# Efficiency only
python3 report.py --efficiency results/efficient/baseline/<id>.json

# Efficiency + accuracy
python3 report.py \
    --efficiency results/efficient/baseline/<id>.json \
    --lmms       results/lmms_eval/.../results.json \
    --output_tag official_v1

# Full comparison (baseline vs TRT, with accuracy)
python3 report.py \
    --efficiency results/efficient/baseline/<id>.json \
    --trt        results/efficient/trt/<id>.json \
    --lmms       results/lmms_eval/.../results.json \
    --output_tag official_v1
```

`--trt` accepts one or more JSON paths for multi-precision comparison (e.g., bf16 + fp8).

**Outputs** (saved to `results/reports/`):

```
report_<run_id>[_<tag>].md
latency_dist_<run_id>.png
vram_breakdown_<run_id>.png
per_sample_latency_<run_id>.png
comparison_<run_id>.png          (only when ≥2 backends or lmms results present)
```

---

## File Listing

```
benchmark/
├── config.py             # paths, model settings, benchmark hyperparameters
├── data_loader.py        # VQAv2 streaming loader (HuggingFace, seeded)
├── metrics.py            # LatencyTimer, VRAM measurement, aggregate statistics
├── run_benchmark.py      # Tool 1: PyTorch baseline efficiency benchmark
├── run_benchmark_trt.py  # Tool 1b: TensorRT-LLM efficiency benchmark
├── report.py             # Tool 3: charts + combined Markdown report
└── README.md
```

---

## Results Directory Layout

```
results/
├── efficient/
│   ├── baseline/         # JSON outputs from run_benchmark.py
│   └── trt/              # JSON outputs from run_benchmark_trt.py
├── lmms_eval/            # lmms-eval output directory
└── reports/              # generated .md reports and .png charts
```

---

## metrics.py API

`metrics.py` provides efficiency measurement utilities only. Do not add accuracy logic here.

| Symbol | Purpose |
|---|---|
| `LatencyTimer` | Records TTFT and total latency via `start()` / `mark_first_token()` / `stop()` |
| `reset_vram_stats()` | Resets PyTorch peak memory counter before each inference |
| `measure_static_vram_gb()` | Reads `memory_allocated()` after model load |
| `measure_dynamic_vram_gb(static_vram_gb)` | Reads `max_memory_allocated()` minus static baseline |
| `compute_decode_latency_per_token(...)` | `(total_ms - ttft_ms) / output_tokens` |
| `aggregate(values)` | Returns `{mean, std, median, p95, min, max}` for a list of floats |
| `summarize_results(per_sample)` | Calls `aggregate()` over all metric keys |

---

*Swift-VLM-Flow — Edge Deployment (Kevin) — CSE 599S, University of Washington*
