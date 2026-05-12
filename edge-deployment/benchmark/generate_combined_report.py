"""
generate_combined_report.py
Combines efficiency benchmark results (from run_benchmark.py) with
accuracy results (from lmms-eval) into a single markdown report.

Usage:
    # Auto-detect latest files
    python3 generate_combined_report.py

    # Specify files explicitly
    python3 generate_combined_report.py \
        --efficiency results/raw/baseline_20260509_030136.json \
        --lmms      results/lmms_eval/models__Qwen2-VL-2B-Instruct/20260512_235305_results.json \
        --output_tag official_v1
"""

import argparse
import json
import os
import glob
from datetime import datetime
import zoneinfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config


# ── File discovery ─────────────────────────────────────────────────────────────

def find_latest(pattern: str) -> str:
    """Return the most recently modified file matching a glob pattern."""
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")
    return max(files, key=os.path.getmtime)


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_efficiency(path: str) -> dict:
    """Load and validate an efficiency benchmark JSON."""
    with open(path) as f:
        data = json.load(f)
    required = {"run_id", "summary", "config", "model_vram_gb"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Efficiency JSON missing keys: {missing}")
    return data


def load_lmms(path: str) -> dict:
    """
    Load and parse a lmms-eval results JSON.

    Returns a simplified dict:
        {
            "task": str,
            "exact_match": float,
            "exact_match_stderr": float,
            "n_samples": int,
            "max_new_tokens": int,
            "post_prompt": str,
            "date": str,
            "eval_time_seconds": float,
        }
    """
    with open(path) as f:
        raw = json.load(f)

    # Find first task (usually only one)
    task_name = list(raw["results"].keys())[0]
    task_result = raw["results"][task_name]
    task_config = raw["configs"][task_name]
    n_samples   = raw["n-samples"][task_name]["effective"]

    return {
        "task":               task_name,
        "exact_match":        task_result.get("exact_match,none", 0.0),
        "exact_match_stderr": task_result.get("exact_match_stderr,none", 0.0),
        "n_samples":          n_samples,
        "max_new_tokens":     task_config.get("generation_kwargs", {}).get("max_new_tokens", "?"),
        "post_prompt":        task_config.get("lmms_eval_specific_kwargs", {}).get("post_prompt", ""),
        "date":               raw.get("date", "unknown"),
        "eval_time_seconds":  float(raw.get("total_evaluation_time_seconds", 0)),
        "lmms_eval_version":  raw.get("lmms_eval_version", "unknown"),
    }


# ── Charts ─────────────────────────────────────────────────────────────────────

def plot_combined_summary(eff: dict, acc: dict, run_id: str) -> str:
    """
    Bar chart comparing key metrics across precision levels.
    Fills in baseline values; TRT columns left as placeholders.
    """
    summary = eff["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Baseline Summary — {config.MODEL_NAME} (fp16 PyTorch)\n"
        f"TRT fp16 and TRT fp8 to be filled after TensorRT conversion",
        fontsize=11
    )

    labels   = ["PyTorch fp16", "TRT fp16", "TRT fp8"]
    colors   = ["#4C72B0", "#cccccc", "#cccccc"]
    hatches  = ["", "//", "//"]

    # Throughput
    tp_vals = [summary["throughput_tok_per_sec"]["mean"], 0, 0]
    bars = axes[0].bar(labels, tp_vals, color=colors, edgecolor="white")
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[0].set_title("Throughput (tok/s)\nhigher is better")
    axes[0].set_ylabel("tokens/sec")
    axes[0].text(0, tp_vals[0] + 1, f"{tp_vals[0]:.1f}", ha="center", fontsize=10)
    axes[0].text(1, 2, "TBD", ha="center", fontsize=9, color="#888888")
    axes[0].text(2, 2, "TBD", ha="center", fontsize=9, color="#888888")

    # TTFT
    ttft_vals = [summary["ttft_ms"]["mean"], 0, 0]
    bars = axes[1].bar(labels, ttft_vals, color=colors, edgecolor="white")
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[1].set_title("TTFT (ms)\nlower is better")
    axes[1].set_ylabel("milliseconds")
    axes[1].text(0, ttft_vals[0] + 1, f"{ttft_vals[0]:.1f}", ha="center", fontsize=10)
    axes[1].text(1, 2, "TBD", ha="center", fontsize=9, color="#888888")
    axes[1].text(2, 2, "TBD", ha="center", fontsize=9, color="#888888")

    # VQAv2 Accuracy
    acc_vals = [acc["exact_match"] * 100, 0, 0]
    bars = axes[2].bar(labels, acc_vals, color=colors, edgecolor="white")
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[2].set_title("VQAv2 Accuracy (%)\nhigher is better")
    axes[2].set_ylabel("%")
    axes[2].set_ylim(0, 110)
    axes[2].text(0, acc_vals[0] + 1, f"{acc_vals[0]:.1f}%", ha="center", fontsize=10)
    axes[2].text(1, 2, "TBD", ha="center", fontsize=9, color="#888888")
    axes[2].text(2, 2, "TBD", ha="center", fontsize=9, color="#888888")

    plt.tight_layout()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    path = f"{config.REPORTS_DIR}/combined_summary_{run_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Report ─────────────────────────────────────────────────────────────────────

def generate_combined_report(
    efficiency_path: str,
    lmms_path: str,
    output_tag: str = "",
) -> str:
    """
    Generate a combined efficiency + accuracy markdown report.

    Args:
        efficiency_path: Path to run_benchmark.py JSON output.
        lmms_path      : Path to lmms-eval results JSON.
        output_tag     : Optional tag appended to output filename.

    Returns:
        Path to the generated markdown report.
    """
    eff = load_efficiency(efficiency_path)
    acc = load_lmms(lmms_path)

    run_id     = eff["run_id"]
    cfg        = eff["config"]
    summary    = eff["summary"]
    model_vram = eff["model_vram_gb"]

    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # Chart
    chart_path = plot_combined_summary(eff, acc, run_id)

    # Accuracy numbers
    acc_pct    = acc["exact_match"] * 100
    acc_stderr = acc["exact_match_stderr"] * 100

    tag = f"_{output_tag}" if output_tag else ""
    report_path = f"{config.REPORTS_DIR}/combined_{run_id}{tag}.md"

    md = f"""# VLM Baseline Benchmark — Combined Report

**Run ID**: `{run_id}`
**Generated**: {datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")
}
**Type**: Efficiency (run_benchmark.py) + Accuracy (lmms-eval)

---

## 1. Environment

| Item | Value |
|------|-------|
| Model | {cfg['model']} |
| Precision | {cfg['precision']} |
| Backend | {cfg['backend']} |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| CUDA Toolkit | 12.8 |
| TensorRT-LLM Container | 0.21.0 |
| PyTorch | 2.8.0 |
| Transformers | 4.51.3 |
| lmms-eval | {acc['lmms_eval_version']} |

---

## 2. Efficiency Results

*{cfg['num_samples']} VQAv2 samples, {cfg['num_warmup']} warmup runs, seed={cfg['seed']}*

| Metric | Mean | Std | Median | p95 |
|--------|------|-----|--------|-----|
| TTFT (ms) | {summary['ttft_ms']['mean']:.1f} | {summary['ttft_ms']['std']:.1f} | {summary['ttft_ms']['median']:.1f} | {summary['ttft_ms']['p95']:.1f} |
| Total Latency (ms) | {summary['total_latency_ms']['mean']:.1f} | {summary['total_latency_ms']['std']:.1f} | {summary['total_latency_ms']['median']:.1f} | {summary['total_latency_ms']['p95']:.1f} |
| Throughput (tok/s) | {summary['throughput_tok_per_sec']['mean']:.1f} | {summary['throughput_tok_per_sec']['std']:.1f} | {summary['throughput_tok_per_sec']['median']:.1f} | {summary['throughput_tok_per_sec']['p95']:.1f} |
| Peak VRAM (GB) | {summary['peak_vram_gb']['mean']:.2f} | {summary['peak_vram_gb']['std']:.3f} | {summary['peak_vram_gb']['median']:.2f} | {summary['peak_vram_gb']['p95']:.2f} |

**Model VRAM (static)**: {model_vram:.2f} GB
**Prompt**: `{config.USER_PROMPT_TEMPLATE.replace('{question}', '<question>')}`
**min_new_tokens**: {config.MIN_NEW_TOKENS} | **max_new_tokens**: {cfg['max_new_tokens']}

---

## 3. Accuracy Results (lmms-eval)

*Evaluated with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) — the standard VLM evaluation toolkit used in research.*

| Item | Value |
|------|-------|
| Task | `{acc['task']}` |
| Metric | exact_match (ignore_case, ignore_punctuation) |
| Num Samples | {acc['n_samples']} |
| Max New Tokens | {acc['max_new_tokens']} |
| Post Prompt | `{acc['post_prompt'].strip()}` |
| Eval Time | {acc['eval_time_seconds']:.1f}s |

### VQAv2 Exact Match Score

| | Value |
|-|-------|
| **Score** | **{acc_pct:.1f}%** |
| Stderr | ±{acc_stderr:.1f}% |

> lmms-eval uses `exact_match` with case and punctuation normalization.
> The prompt `{acc['post_prompt'].strip()}` is automatically appended to each question,
> instructing the model to answer with a single word or phrase —
> matching the VQAv2 ground truth format.

---

## 4. Summary Chart

![Combined Summary](combined_summary_{run_id}.png)

*Grey bars (TBD) will be filled after TensorRT engine conversion.*

---

## 5. Baseline Numbers — TRT Comparison Table

*Reference table to be completed after TensorRT fp16 and fp8 engine conversion.*

| Metric | PyTorch fp16 | TRT fp16 | TRT fp8 |
|--------|-------------|----------|---------|
| TTFT (ms) | {summary['ttft_ms']['mean']:.1f} | — | — |
| Total Latency (ms) | {summary['total_latency_ms']['mean']:.1f} | — | — |
| Throughput (tok/s) | {summary['throughput_tok_per_sec']['mean']:.1f} | — | — |
| Peak VRAM (GB) | {summary['peak_vram_gb']['mean']:.2f} | — | — |
| Model VRAM (GB) | {model_vram:.2f} | — | — |
| VQAv2 Accuracy | {acc_pct:.1f}% | — | — |
| Speedup vs baseline | 1.0× | — | — |

---

## 6. Methodology

### Efficiency
- TTFT measured via a separate `max_new_tokens=1` generation call.
- Total latency covers generation of up to `{cfg['max_new_tokens']}` tokens (`min_new_tokens={config.MIN_NEW_TOKENS}`).
- All timing uses `torch.cuda.synchronize()` + `time.perf_counter()`.
- Peak VRAM via `torch.cuda.max_memory_allocated()`, reset per sample.
- {cfg['num_warmup']} warmup runs before measurement.

### Accuracy
- Evaluated with **lmms-eval**, the standard toolkit used in VLM compression research
  (MBQ CVPR 2025, LiteVLM, GRACE, etc.).
- Task: `vqav2_val` — VQAv2 validation split.
- Metric: `exact_match` with case and punctuation normalization.
- Prompt: question + `\\nAnswer the question using a single word or phrase.`
- This prompt format matches the VQAv2 ground truth annotation style
  (single words or short phrases).

### Reproducibility
- Efficiency: `random.seed({cfg['seed']})` for VQAv2 subset sampling.
- Accuracy: lmms-eval `random_seed=0`, `torch_seed=1234`.
- Docker image: `nvcr.io/nvidia/tensorrt-llm/release:0.21.0`

---

## 7. Related Work

| Paper | Relevance |
|-------|-----------|
| LiteVLM (arXiv:2506.07416) | Edge VLM deployment, FP8 quantization, 2.5×–3.2× speedup |
| MBQ (CVPR 2025) | Qwen2-VL quantization accuracy evaluation with lmms-eval |
| GRACE (arXiv:2601.22709) | Qwen2-VL-2B INT4 quantization baseline numbers |
| Edge Reliability Gap (arXiv:2603.26769) | VQAv2 evaluation on RTX hardware, compressed VLMs |
| LLMC+ (arXiv:2508.09981) | Comprehensive VLM compression benchmark framework |

---

*Report generated by `generate_combined_report.py` — Swift-VLM-Flow, CSE 599S, UW*
"""

    with open(report_path, "w") as f:
        f.write(md)

    print(f"  Chart  → {chart_path}")
    print(f"  Report → {report_path}")
    return report_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate combined efficiency + accuracy report")
    parser.add_argument("--efficiency", type=str, default=None,
                        help="Path to run_benchmark.py JSON (default: latest in results/raw/)")
    parser.add_argument("--lmms", type=str, default=None,
                        help="Path to lmms-eval results JSON (default: latest in results/lmms_eval/)")
    parser.add_argument("--output_tag", type=str, default="",
                        help="Optional tag appended to output filename")
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect latest files if not specified
    eff_path = args.efficiency or find_latest(
        os.path.join(config.RAW_DIR, "baseline_*.json")
    )
    lmms_path = args.lmms or find_latest(
        os.path.join(config.RESULTS_DIR, "lmms_eval", "**", "*_results.json")
    )

    print("=" * 60)
    print(" Combined Report Generator")
    print(f"  Efficiency : {eff_path}")
    print(f"  lmms-eval  : {lmms_path}")
    print("=" * 60)

    report = generate_combined_report(eff_path, lmms_path, args.output_tag)
    print(f"\nDone. Open: {report}")


if __name__ == "__main__":
    main()
