# Benchmark — Swift-VLM-Flow

Two-tool benchmark design for Qwen2-VL-2B-Instruct evaluation.
Part of the **Swift-VLM-Flow** project (CSE 599S, UW).

---

## Two-Tool Benchmark Design

### Tool 1 — run_benchmark.py (efficiency)

```bash
python3 run_benchmark.py --num_samples 50 --warmup 3
```

Measures: TTFT, latency, throughput, VRAM  
Uses: VQAv2 streaming, `min_new_tokens=20`, proper warmup

### Tool 2 — lmms-eval (accuracy)

```bash
python -m lmms_eval --model qwen2_vl \
  --model_args pretrained=/workspace/models/Qwen2-VL-2B-Instruct \
  --tasks vqav2_val --batch_size 1 --limit 500 \
  --output_path /workspace/benchmark/results/lmms_eval/
```

Measures: VQAv2 exact_match  
Standard research toolkit (MBQ, LiteVLM, GRACE all use this)

### Tool 3 — generate_combined_report.py (report)

```bash
python3 generate_combined_report.py
```

Merges both outputs into one markdown report with charts

---

## File Listing

```
benchmark/
├── config.py                    # all settings
├── data_loader.py               # VQAv2 streaming loader
├── metrics.py                   # TTFT, VRAM, throughput measurement
├── run_benchmark.py             # efficiency benchmark main script
├── report_generator.py          # charts + efficiency markdown report
├── generate_combined_report.py  # merged efficiency + accuracy report
└── README.md
```

---

## Note on metrics.py

`metrics.py` contains only efficiency measurement utilities:
`LatencyTimer`, `measure_vram_gb`, `reset_vram_stats`, `compute_throughput`,
`aggregate`, `summarize_results`.

VQAv2 accuracy evaluation is handled entirely by lmms-eval —
do not add custom accuracy logic here.

---

*Swift-VLM-Flow — Edge Deployment (Kevin) — CSE 599S, University of Washington*
