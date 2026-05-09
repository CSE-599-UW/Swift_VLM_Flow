"""
report_generator.py
Generates a Markdown report and matplotlib charts from a raw benchmark JSON file.

Outputs (saved to config.REPORTS_DIR):
  - baseline_<run_id>.md          : Full markdown report
  - latency_distribution_<run_id>.png  : TTFT + Total Latency histograms
  - throughput_vram_<run_id>.png       : Throughput + VRAM scatter / box plots
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works inside Docker)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import config


# ── Chart Generation ───────────────────────────────────────────────────────────

def plot_latency_distribution(per_sample: list, run_id: str) -> str:
    """Histogram of TTFT and Total Latency across all samples."""
    ttft    = [s["ttft_ms"] for s in per_sample]
    latency = [s["total_latency_ms"] for s in per_sample]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Latency Distribution — {config.MODEL_NAME} (fp16 PyTorch)", fontsize=13)

    axes[0].hist(ttft, bins=20, color="#4C72B0", edgecolor="white")
    axes[0].axvline(np.mean(ttft), color="red", linestyle="--", label=f"Mean={np.mean(ttft):.1f}ms")
    axes[0].axvline(np.percentile(ttft, 95), color="orange", linestyle="--",
                    label=f"p95={np.percentile(ttft, 95):.1f}ms")
    axes[0].set_title("Time to First Token (TTFT)")
    axes[0].set_xlabel("ms")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(latency, bins=20, color="#55A868", edgecolor="white")
    axes[1].axvline(np.mean(latency), color="red", linestyle="--",
                    label=f"Mean={np.mean(latency):.1f}ms")
    axes[1].axvline(np.percentile(latency, 95), color="orange", linestyle="--",
                    label=f"p95={np.percentile(latency, 95):.1f}ms")
    axes[1].set_title("Total Latency")
    axes[1].set_xlabel("ms")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    path = f"{config.REPORTS_DIR}/latency_distribution_{run_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_throughput_vram(per_sample: list, run_id: str) -> str:
    """Box plot of Throughput and VRAM across all samples."""
    throughput = [s["throughput_tok_per_sec"] for s in per_sample]
    vram       = [s["peak_vram_gb"] for s in per_sample]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Throughput & VRAM — {config.MODEL_NAME} (fp16 PyTorch)", fontsize=13)

    axes[0].boxplot(throughput, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.7))
    axes[0].set_title("Throughput (tokens/sec)")
    axes[0].set_ylabel("tokens/sec")
    axes[0].set_xticks([1])
    axes[0].set_xticklabels([config.MODEL_NAME])

    axes[1].boxplot(vram, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#C44E52", alpha=0.7))
    axes[1].set_title("Peak VRAM (GB)")
    axes[1].set_ylabel("GB")
    axes[1].set_xticks([1])
    axes[1].set_xticklabels([config.MODEL_NAME])

    plt.tight_layout()
    path = f"{config.REPORTS_DIR}/throughput_vram_{run_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_per_sample_latency(per_sample: list, run_id: str) -> str:
    """Line chart of total latency per sample (shows variance over the run)."""
    latencies = [s["total_latency_ms"] for s in per_sample]
    indices   = list(range(1, len(latencies) + 1))
    mean_val  = np.mean(latencies)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(indices, latencies, color="#4C72B0", linewidth=0.8, alpha=0.8, label="Latency")
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_val:.1f} ms")
    ax.fill_between(indices, latencies, mean_val, alpha=0.1, color="#4C72B0")
    ax.set_title(f"Per-sample Total Latency — {config.MODEL_NAME} (fp16 PyTorch)", fontsize=13)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Latency (ms)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = f"{config.REPORTS_DIR}/per_sample_latency_{run_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Markdown Report ────────────────────────────────────────────────────────────

def _stats_table(summary: dict) -> str:
    """Render the summary statistics as a markdown table."""
    header = "| Metric | Mean | Std | Median | p95 | Min | Max |\n"
    sep    = "|--------|------|-----|--------|-----|-----|-----|\n"
    rows   = ""

    labels = {
        "ttft_ms":                "TTFT (ms)",
        "total_latency_ms":       "Total Latency (ms)",
        "throughput_tok_per_sec": "Throughput (tok/s)",
        "peak_vram_gb":           "Peak VRAM (GB)",
        "output_tokens":          "Output Tokens",
    }

    for key, label in labels.items():
        if key in summary:
            s = summary[key]
            rows += (
                f"| {label} "
                f"| {s['mean']} "
                f"| {s['std']} "
                f"| {s['median']} "
                f"| {s['p95']} "
                f"| {s['min']} "
                f"| {s['max']} |\n"
            )
    return header + sep + rows


def generate_report(raw_json_path: str) -> str:
    """
    Generate a markdown report and charts from a raw benchmark JSON file.

    Args:
        raw_json_path: Path to the JSON file produced by run_benchmark.py.

    Returns:
        Path to the generated markdown report.
    """
    with open(raw_json_path) as f:
        data = json.load(f)

    run_id     = data["run_id"]
    cfg        = data["config"]
    summary    = data["summary"]
    per_sample = data["per_sample"]
    model_vram = data["model_vram_gb"]

    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # Generate charts
    chart_latency   = plot_latency_distribution(per_sample, run_id)
    chart_tp_vram   = plot_throughput_vram(per_sample, run_id)
    chart_per_sample = plot_per_sample_latency(per_sample, run_id)

    # Build markdown
    md = f"""# VLM Baseline Benchmark Report

**Run ID**: `{run_id}`  
**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Environment

| Item | Value |
|------|-------|
| Model | {cfg['model']} |
| Precision | {cfg['precision']} |
| Backend | {cfg['backend']} |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| CUDA Toolkit | 12.8 |
| TensorRT-LLM Container | 0.21.0 (nvcr.io/nvidia/tensorrt-llm/release:0.21.0) |
| PyTorch | 2.8.0 |
| Transformers | 4.51.3 |

---

## 2. Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | {cfg['dataset']} ({cfg['split']}) |
| Num Samples | {cfg['num_samples']} |
| Num Warmup Runs | {cfg['num_warmup']} |
| Max New Tokens | {cfg['max_new_tokens']} |
| Batch Size | 1 |
| Random Seed | {cfg['seed']} |
| Prompt Format | `{{question}} Answer briefly.` |

---

## 3. Results Summary

{_stats_table(summary)}

**Model VRAM (static load)**: {model_vram:.2f} GB

### Key Observations

- **TTFT**: Mean TTFT of **{summary['ttft_ms']['mean']:.1f} ms** (p95: {summary['ttft_ms']['p95']:.1f} ms).
  This measures the prefill cost including visual token encoding.
- **Throughput**: Mean throughput of **{summary['throughput_tok_per_sec']['mean']:.1f} tok/s**,
  indicating the raw decoding speed of the fp16 PyTorch baseline.
- **VRAM**: Peak inference VRAM of **{summary['peak_vram_gb']['mean']:.2f} GB**
  on top of the {model_vram:.2f} GB model footprint.

---

## 4. Charts

### 4.1 Latency Distribution
![Latency Distribution](latency_distribution_{run_id}.png)

### 4.2 Throughput & VRAM
![Throughput and VRAM](throughput_vram_{run_id}.png)

### 4.3 Per-sample Latency
![Per-sample Latency](per_sample_latency_{run_id}.png)

---

## 5. Methodology

### Measurement Approach

- **TTFT** is measured by running a separate `max_new_tokens=1` generation call
  and recording the time from input preparation to the first decoded token.
- **Total Latency** covers the complete generation of up to `{cfg['max_new_tokens']}` tokens.
- **Throughput** is computed as `output_tokens / total_latency_seconds`.
- **Peak VRAM** is measured using `torch.cuda.max_memory_allocated()` with stats
  reset before each sample.
- All timing uses `torch.cuda.synchronize()` + `time.perf_counter()` to ensure
  GPU operations are fully complete before recording timestamps.
- {cfg['num_warmup']} warmup runs are performed before measurement to eliminate
  JIT compilation and caching overhead.

### Reproducibility

The VQAv2 subset is drawn using `random.seed({cfg['seed']})` for reproducibility.
The same seed and sample indices must be used when comparing against TRT results.

---

## 6. Baseline Numbers (Reference for TRT Comparison)

These numbers serve as the reference baseline for measuring TensorRT optimization gains.

| Metric | Baseline (fp16 PyTorch) | TRT fp16 | TRT fp8 |
|--------|------------------------|----------|---------|
| TTFT (ms) | {summary['ttft_ms']['mean']:.1f} | — | — |
| Total Latency (ms) | {summary['total_latency_ms']['mean']:.1f} | — | — |
| Throughput (tok/s) | {summary['throughput_tok_per_sec']['mean']:.1f} | — | — |
| Peak VRAM (GB) | {summary['peak_vram_gb']['mean']:.2f} | — | — |
| Model VRAM (GB) | {model_vram:.2f} | — | — |

*TRT columns to be filled after TensorRT engine conversion.*

---

*Report generated by `report_generator.py` — Swift-VLM-Flow Project, CSE 599S, UW*
"""

    report_path = f"{config.REPORTS_DIR}/baseline_{run_id}.md"
    with open(report_path, "w") as f:
        f.write(md)

    return report_path
