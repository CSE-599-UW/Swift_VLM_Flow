"""
report.py
Generates efficiency + accuracy charts and a combined Markdown report
for the Swift-VLM-Flow benchmark pipeline.

Usage:
    # Efficiency only (lmms-eval not yet run)
    python3 report.py --efficiency results/raw/baseline_<id>.json

    # Full combined report
    python3 report.py \\
        --efficiency results/raw/baseline_<id>.json \\
        --lmms       results/lmms_eval/.../results.json

    # Multi-backend comparison
    python3 report.py \\
        --efficiency results/raw/baseline_<id>.json \\
        --trt        results/raw/trt_<id>.json [results/raw/trt_fp8_<id>.json ...] \\
        --lmms       results/lmms_eval/.../results.json \\
        --output_tag official_v1

Inputs
------
Efficiency JSON (from run_benchmark.py):
    {
      "run_id":         str,
      "timestamp":      str,
      "config":         { model, precision, backend, num_samples, num_warmup,
                          max_new_tokens, min_new_tokens, dataset, split, seed },
      "static_vram_gb": float,
      "summary": {
        "ttft_ms":                   { mean, std, median, p95, min, max },
        "decode_latency_ms_per_tok": { mean, std, median, p95, min, max },
        "dynamic_vram_gb":           { mean, std, median, p95, min, max },
        "output_tokens":             { mean, std, median, p95, min, max },
      },
      "per_sample": [ { question_id, question, predicted_answer,
                        ground_truth_answers, ttft_ms, total_latency_ms,
                        decode_latency_ms_per_tok, dynamic_vram_gb,
                        output_tokens }, ... ]
    }

lmms-eval JSON:  standard lmms-eval results format.
    Supported tasks: vqav2_val, pope, mme (others silently ignored).

Outputs (saved to config.REPORTS_DIR)
--------------------------------------
    report_<run_id>[_<tag>].md
    latency_dist_<run_id>.png
    vram_breakdown_<run_id>.png
    per_sample_latency_<run_id>.png
    comparison_<run_id>.png          (only when ≥2 efficiency runs or lmms present)
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


# ── Compatibility shim ─────────────────────────────────────────────────────────

def _compat_load(data: dict) -> dict:
    """
    Normalise legacy JSON (produced before experiment_pipeline_design.md was
    finalised) to the current schema.

    Legacy keys handled:
      model_vram_gb          → static_vram_gb
      summary.peak_vram_gb   → summary.dynamic_vram_gb
      summary.throughput_tok_per_sec → summary.decode_latency_ms_per_tok  (approx)
    """
    if "static_vram_gb" not in data:
        data["static_vram_gb"] = data.get("model_vram_gb", 0.0)

    s = data["summary"]

    if "dynamic_vram_gb" not in s:
        s["dynamic_vram_gb"] = s.get("peak_vram_gb", {
            "mean": 0, "std": 0, "median": 0, "p95": 0, "min": 0, "max": 0
        })

    if "decode_latency_ms_per_tok" not in s:
        if "throughput_tok_per_sec" in s:
            tp = s["throughput_tok_per_sec"]
            def inv(v): return round(1000.0 / v, 3) if v and v > 0 else 0
            s["decode_latency_ms_per_tok"] = {
                "mean": inv(tp.get("mean", 0)), "std": 0,
                "median": inv(tp.get("median", 0)),
                "p95":  inv(tp.get("p95", 0)),
                "min":  inv(tp.get("max", 0)),
                "max":  inv(tp.get("min", 0)),
            }
        else:
            s["decode_latency_ms_per_tok"] = {
                "mean": 0, "std": 0, "median": 0, "p95": 0, "min": 0, "max": 0
            }

    for sample in data.get("per_sample", []):
        if "decode_latency_ms_per_tok" not in sample:
            tok   = sample.get("output_tokens", 0) or 0
            total = sample.get("total_latency_ms", 0) or 0
            ttft  = sample.get("ttft_ms", 0) or 0
            sample["decode_latency_ms_per_tok"] = (
                round((total - ttft) / tok, 3) if tok > 0 else 0
            )
        if "dynamic_vram_gb" not in sample:
            sample["dynamic_vram_gb"] = sample.get("peak_vram_gb", 0)

    return data


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_efficiency(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    for key in ("run_id", "summary", "config"):
        if key not in data:
            raise ValueError(f"Efficiency JSON missing key: '{key}' in {path}")
    return _compat_load(data)


def _load_lmms(path: str) -> dict | None:
    """
    Parse a lmms-eval results JSON.
    Returns None if path is None; raises on malformed files.

    Supported task prefixes: vqav2, pope, mme.
    Unknown tasks are silently ignored.

    Return schema:
        {
          "tasks": {
            "vqav2": { task, score, stderr, n_samples, max_new_tokens, post_prompt } | None,
            "pope":  { task, accuracy, f1, n_samples } | None,
            "mme":   { task, perception, cognition, total, n_samples } | None,
          },
          "date": str,
          "eval_time_seconds": float,
          "lmms_eval_version": str,
        }
    """
    if path is None:
        return None

    with open(path) as f:
        raw = json.load(f)

    results = raw.get("results", {})
    configs = raw.get("configs", {})
    n_samp  = raw.get("n-samples", {})

    out = {
        "tasks": {"vqav2": None, "pope": None, "mme": None},
        "date": raw.get("date", "unknown"),
        "eval_time_seconds": float(raw.get("total_evaluation_time_seconds", 0)),
        "lmms_eval_version": raw.get("lmms_eval_version", "unknown"),
    }

    for task_key, task_res in results.items():
        tk = task_key.lower()
        cfg_t = configs.get(task_key, {})

        if "vqav2" in tk:
            out["tasks"]["vqav2"] = {
                "task":           task_key,
                "score":          task_res.get("exact_match,none", 0.0),
                "stderr":         task_res.get("exact_match_stderr,none", 0.0),
                "n_samples":      n_samp.get(task_key, {}).get("effective", "?"),
                "max_new_tokens": cfg_t.get("generation_kwargs", {}).get("max_new_tokens", "?"),
                "post_prompt":    cfg_t.get("lmms_eval_specific_kwargs", {}).get("post_prompt", ""),
            }
        elif "pope" in tk:
            out["tasks"]["pope"] = {
                "task":      task_key,
                "accuracy":  task_res.get("pope_accuracy,none",
                             task_res.get("accuracy,none", 0.0)),
                "f1":        task_res.get("pope_f1,none",
                             task_res.get("f1,none", 0.0)),
                "n_samples": n_samp.get(task_key, {}).get("effective", "?"),
            }
        elif "mme" in tk:
            perception = task_res.get("mme_percetion_score,none",
                         task_res.get("perception_score,none",
                         task_res.get("score,none", 0.0)))
            cognition  = task_res.get("mme_cognition_score,none",
                         task_res.get("cognition_score,none", 0.0))
            out["tasks"]["mme"] = {
                "task":       task_key,
                "perception": perception,
                "cognition":  cognition,
                "total":      perception + cognition,
                "n_samples":  n_samp.get(task_key, {}).get("effective", "?"),
            }

    return out


# ── Helpers ────────────────────────────────────────────────────────────────────

def _backend_label(cfg: dict) -> str:
    return f"{cfg.get('model', '')} · {cfg.get('backend', '')} · {cfg.get('precision', '')}"


def _fmt(val, d=1):
    try:
        return f"{val:.{d}f}"
    except (TypeError, ValueError):
        return str(val)


def _speedup(base: float, new: float) -> str:
    """Latency speedup (lower-is-better): base/new. Arrow shows direction."""
    if not base or not new:
        return "—"
    ratio = base / new
    return f"{'▲' if ratio > 1 else '▼'} {ratio:.2f}×"


def _find_latest(pattern: str) -> str:
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return max(files, key=os.path.getmtime)


# ── Charts ─────────────────────────────────────────────────────────────────────

def _plot_latency_distributions(per_sample: list, cfg: dict, run_id: str) -> str:
    """TTFT and Decode Latency per token histograms."""
    ttft    = [s["ttft_ms"]                   for s in per_sample]
    dec_lat = [s["decode_latency_ms_per_tok"] for s in per_sample]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Latency Distribution — {_backend_label(cfg)}", fontsize=12)

    for ax, vals, title, xlabel, color in [
        (axes[0], ttft,    "TTFT — Prefill Phase\n(time to first token)",      "ms",       "#4C72B0"),
        (axes[1], dec_lat, "Decode Latency — Decode Phase\n(ms per output token)", "ms/tok", "#55A868"),
    ]:
        ax.hist(vals, bins=20, color=color, edgecolor="white")
        ax.axvline(np.mean(vals), color="red", linestyle="--",
                   label=f"Mean = {np.mean(vals):.1f}")
        ax.axvline(np.percentile(vals, 95), color="orange", linestyle="--",
                   label=f"p95  = {np.percentile(vals, 95):.1f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    path = f"{config.REPORTS_DIR}/latency_dist_{run_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _plot_vram_breakdown(static_vram: float, per_sample: list,
                         cfg: dict, run_id: str) -> str:
    """Stacked bar (static + dynamic) and dynamic box plot."""
    dynamic = [s["dynamic_vram_gb"] for s in per_sample]
    n       = len(dynamic)
    static_arr = np.full(n, static_vram)
    sort_idx   = np.argsort(dynamic)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"VRAM Breakdown — {_backend_label(cfg)}", fontsize=12)

    dyn_sorted = np.array(dynamic)[sort_idx]
    axes[0].bar(range(n), static_arr,
                label=f"Static ({static_vram:.2f} GB fixed)", color="#4C72B0", alpha=0.85)
    axes[0].bar(range(n), dyn_sorted, bottom=static_arr,
                label="Dynamic (per-inference)", color="#C44E52", alpha=0.85)
    axes[0].axhline(static_vram + np.mean(dynamic), color="black", linestyle="--",
                    linewidth=1, label=f"Mean total = {static_vram + np.mean(dynamic):.3f} GB")
    axes[0].set_title("Static + Dynamic VRAM per Sample\n(sorted by dynamic)")
    axes[0].set_xlabel("Sample (sorted)")
    axes[0].set_ylabel("GB")
    axes[0].legend(fontsize=8)

    axes[1].boxplot(dynamic, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#C44E52", alpha=0.7))
    axes[1].set_title(f"Dynamic VRAM Distribution\nmean = {np.mean(dynamic):.3f} GB")
    axes[1].set_ylabel("GB")
    axes[1].set_xticks([1])
    axes[1].set_xticklabels([cfg.get("backend", "")])

    plt.tight_layout()
    path = f"{config.REPORTS_DIR}/vram_breakdown_{run_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _plot_per_sample_latency(per_sample: list, cfg: dict, run_id: str) -> str:
    """TTFT and Decode Latency per token line charts across samples."""
    ttft    = [s["ttft_ms"]                   for s in per_sample]
    dec_lat = [s["decode_latency_ms_per_tok"] for s in per_sample]
    idx     = list(range(1, len(ttft) + 1))

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(f"Per-sample Latency — {_backend_label(cfg)}", fontsize=12)

    for ax, vals, ylabel, color in [
        (axes[0], ttft,    "TTFT (ms)",              "#4C72B0"),
        (axes[1], dec_lat, "Decode Latency (ms/tok)", "#55A868"),
    ]:
        mean_v = np.mean(vals)
        ax.plot(idx, vals, color=color, linewidth=0.9, alpha=0.8)
        ax.axhline(mean_v, color="red", linestyle="--", linewidth=1.2,
                   label=f"Mean = {mean_v:.1f}")
        ax.fill_between(idx, vals, mean_v, alpha=0.1, color=color)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].set_xlabel("Sample index")
    plt.tight_layout()
    path = f"{config.REPORTS_DIR}/per_sample_latency_{run_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _plot_comparison(eff_list: list, vqav2_score: float | None, run_id: str) -> str:
    """
    4-panel comparison bar chart across backends:
      TTFT · Decode Latency · Static VRAM · VQAv2 Accuracy
    Missing runs shown as grey hatched placeholders.
    """
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    grey    = "#cccccc"
    n       = len(eff_list)

    labels = [
        e["config"]["backend"] + "\n" + e["config"]["precision"]
        for e in eff_list
    ]
    # If only one run, add TBD placeholders
    plot_labels = labels if n > 1 else labels + ["TRT fp16\n(TBD)", "TRT fp8\n(TBD)"]
    n_bars = len(plot_labels)

    def make_vals(getter):
        return [getter(e) for e in eff_list] + [0] * (n_bars - n)

    panels = [
        ("TTFT · prefill (ms)\n↓ better",
         make_vals(lambda e: e["summary"]["ttft_ms"]["mean"])),
        ("Decode Latency (ms/tok)\n↓ better",
         make_vals(lambda e: e["summary"]["decode_latency_ms_per_tok"]["mean"])),
        ("Static VRAM (GB)\n↓ better",
         make_vals(lambda e: e["static_vram_gb"])),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Swift-VLM-Flow — {config.MODEL_NAME} Backend Comparison\n"
        "Speed ↓ · Memory ↓ · Accuracy ↑",
        fontsize=11,
    )

    for ax, (title, vals) in zip(axes[:3], panels):
        colors = [palette[i % len(palette)] for i in range(n)] + [grey] * (n_bars - n)
        bars = ax.bar(plot_labels, vals, color=colors, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v == 0:
                bar.set_hatch("//")
                ax.text(bar.get_x() + bar.get_width() / 2,
                        (max(vals) or 1) * 0.03,
                        "TBD", ha="center", fontsize=8, color="#888888")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v * 1.02, f"{v:.1f}", ha="center", fontsize=9)
        ax.set_title(title, fontsize=10)

    # Accuracy panel
    ax = axes[3]
    acc_val = (vqav2_score * 100) if vqav2_score is not None else 0
    acc_vals   = [acc_val] + [0] * (n_bars - 1)
    acc_colors = [palette[0]] + [grey] * (n_bars - 1)
    bars = ax.bar(plot_labels, acc_vals, color=acc_colors, edgecolor="white")
    for bar, v in zip(bars, acc_vals):
        if v == 0:
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2, 0.5,
                    "TBD", ha="center", fontsize=8, color="#888888")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_title("VQAv2 Accuracy (%)\n↑ better", fontsize=10)

    plt.tight_layout()
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    path = f"{config.REPORTS_DIR}/comparison_{run_id}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ── Markdown section builders ──────────────────────────────────────────────────

def _stats_table(summary: dict, static_vram: float) -> str:
    header = "| Metric | Mean | Std | Median | p95 | Min | Max |\n"
    sep    = "|--------|------|-----|--------|-----|-----|-----|\n"

    def row(label, s, d=1):
        return (
            f"| {label} "
            f"| {_fmt(s.get('mean','—'),d)} "
            f"| {_fmt(s.get('std','—'),d)} "
            f"| {_fmt(s.get('median','—'),d)} "
            f"| {_fmt(s.get('p95','—'),d)} "
            f"| {_fmt(s.get('min','—'),d)} "
            f"| {_fmt(s.get('max','—'),d)} |\n"
        )

    rows  = "| **Speed** | | | | | | |\n"
    rows += row("TTFT · prefill (ms)",       summary["ttft_ms"])
    rows += row("Decode Latency (ms/tok)",   summary["decode_latency_ms_per_tok"])
    rows += "| **Memory** | | | | | | |\n"
    rows += f"| Static VRAM (GB) | {static_vram:.3f} | — | — | — | — | — |\n"
    rows += row("Dynamic VRAM (GB)",         summary["dynamic_vram_gb"], d=3)
    rows += "| **Output** | | | | | | |\n"
    rows += row("Output Tokens",             summary["output_tokens"])

    return header + sep + rows


def _accuracy_section(acc: dict) -> str:
    tasks = acc["tasks"]
    parts = []

    vqa = tasks.get("vqav2")
    if vqa:
        parts.append(f"""### VQAv2 Exact Match

| Item | Value |
|------|-------|
| Task | `{vqa['task']}` |
| Metric | exact_match (ignore_case, ignore_punctuation) |
| Samples | {vqa['n_samples']} |
| Max New Tokens | {vqa['max_new_tokens']} |
| Post Prompt | `{str(vqa['post_prompt']).strip()}` |

**Score: {vqa['score']*100:.1f}% ± {vqa['stderr']*100:.1f}%**
""")
    else:
        parts.append("### VQAv2\n*Not evaluated in this lmms-eval run.*\n")

    pope = tasks.get("pope")
    if pope:
        parts.append(f"""### POPE (Hallucination)

| Metric | Value |
|--------|-------|
| Accuracy | {pope['accuracy']*100:.1f}% |
| F1 | {pope['f1']*100:.1f}% |
| Samples | {pope['n_samples']} |

> POPE probes object hallucination — the most common failure mode of
> compressed VLMs. F1 is the primary metric.
""")
    else:
        parts.append("### POPE\n*Not evaluated — run `lmms_eval --tasks pope` to add.*\n")

    mme = tasks.get("mme")
    if mme:
        parts.append(f"""### MME (Perception + Cognition)

| Sub-score | Value |
|-----------|-------|
| Perception | {mme['perception']:.1f} |
| Cognition  | {mme['cognition']:.1f} |
| **Total**  | **{mme['total']:.1f}** |

> MME uses choice format — scores are unaffected by answer length,
> making it stable across backends.
""")
    else:
        parts.append("### MME\n*Not evaluated — run `lmms_eval --tasks mme` to add.*\n")

    return "\n".join(parts)


def _comparison_table(eff_list: list, acc: dict | None) -> str:
    if len(eff_list) < 2:
        return ""

    baseline = eff_list[0]
    others   = eff_list[1:]

    header = ("| Metric | "
              + f"{baseline['config']['backend']} {baseline['config']['precision']} (baseline) | "
              + "".join(f"{e['config']['backend']} {e['config']['precision']} | " for e in others)
              + "".join(f"vs {e['config']['backend']} | " for e in others)
              + "\n")
    sep = "|--------|" + "--------|" * (1 + len(others) * 2) + "\n"

    def eff_row(label, getter, d=1):
        base_val = getter(baseline)
        row = f"| {label} | {_fmt(base_val, d)} | "
        row += "".join(f"{_fmt(getter(e), d)} | " for e in others)
        row += "".join(f"{_speedup(base_val, getter(e))} | " for e in others)
        return row + "\n"

    rows  = eff_row("TTFT · prefill (ms)",     lambda e: e["summary"]["ttft_ms"]["mean"])
    rows += eff_row("Decode Latency (ms/tok)",  lambda e: e["summary"]["decode_latency_ms_per_tok"]["mean"])
    rows += eff_row("Static VRAM (GB)",         lambda e: e["static_vram_gb"], d=3)
    rows += eff_row("Dynamic VRAM (GB)",        lambda e: e["summary"]["dynamic_vram_gb"]["mean"], d=3)

    if acc and acc["tasks"].get("vqav2"):
        score = acc["tasks"]["vqav2"]["score"] * 100
        rows += f"| VQAv2 Accuracy | {score:.1f}% | " + "— | " * len(others) * 2 + "\n"

    note = "\n*▲ speedup > 1× = faster than baseline (latency metrics, lower is better)*\n"
    return header + sep + rows + note


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_report(
    efficiency_paths: list[str],
    lmms_path: str | None = None,
    output_tag: str = "",
) -> str:
    """
    Generate charts and a Markdown report.

    Args:
        efficiency_paths: Paths to run_benchmark.py JSONs.
                          First element = baseline; rest = TRT variants.
        lmms_path:        Path to lmms-eval JSON, or None (efficiency-only mode).
        output_tag:       Optional suffix for the output filename.

    Returns:
        Path to the generated .md report.
    """
    eff_list = [_load_efficiency(p) for p in efficiency_paths]
    acc      = _load_lmms(lmms_path)

    baseline    = eff_list[0]
    run_id      = baseline["run_id"]
    cfg         = baseline["config"]
    summary     = baseline["summary"]
    per_sample  = baseline["per_sample"]
    static_vram = baseline["static_vram_gb"]

    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # Charts
    chart_latency    = _plot_latency_distributions(per_sample, cfg, run_id)
    chart_vram       = _plot_vram_breakdown(static_vram, per_sample, cfg, run_id)
    chart_per_sample = _plot_per_sample_latency(per_sample, cfg, run_id)

    vqav2_score  = acc["tasks"]["vqav2"]["score"] if (acc and acc["tasks"].get("vqav2")) else None
    show_compare = len(eff_list) > 1 or acc is not None
    chart_compare = (
        _plot_comparison(eff_list, vqav2_score, run_id) if show_compare else None
    )

    # Shorthand stats
    ttft_s    = summary["ttft_ms"]
    dec_s     = summary["decode_latency_ms_per_tok"]
    dyn_s     = summary["dynamic_vram_gb"]
    total_vram_mean = static_vram + dyn_s["mean"]

    now_pst = datetime.now(
        zoneinfo.ZoneInfo("America/Los_Angeles")
    ).strftime("%Y-%m-%d %H:%M:%S %Z")

    mode_note = (
        "Efficiency + Accuracy (lmms-eval)" if acc
        else "Efficiency only (no lmms-eval provided)"
    )

    # Env rows for all runs
    env_runs = "\n".join(
        f"| {'Baseline' if i == 0 else f'TRT #{i}'} | `{e['run_id']}` "
        f"| {e['config']['backend']} | {e['config']['precision']} |"
        for i, e in enumerate(eff_list)
    )

    # Accuracy block
    acc_block = (
        f"""---

## 4. Accuracy Results (lmms-eval)

*Eval time: {acc['eval_time_seconds']:.0f}s · Date: {acc['date']} · lmms-eval {acc['lmms_eval_version']}*

{_accuracy_section(acc)}"""
        if acc else
        """---

## 4. Accuracy Results

*lmms-eval not provided. Run with `--lmms <path>` to include accuracy.*"""
    )

    # Comparison block
    comp_block = (
        f"""---

## 5. Cross-Backend Comparison

{_comparison_table(eff_list, acc)}
![Comparison Chart](comparison_{run_id}.png)"""
        if show_compare else ""
    )

    section_num = 6 if show_compare else 5

    md = f"""# VLM Benchmark Report

**Run ID**: `{run_id}`
**Generated**: {now_pst}
**Mode**: {mode_note}

---

## 1. Environment

| Item | Value |
|------|-------|
| Model | {cfg['model']} |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| CUDA Toolkit | 12.8 |
| TensorRT-LLM Container | 0.21.0 |
| PyTorch | 2.8.0 |
| Transformers | 4.51.3 |{f"""
| lmms-eval | {acc['lmms_eval_version']} |""" if acc else ""}

### Runs

| Role | Run ID | Backend | Precision |
|------|--------|---------|-----------|
{env_runs}

---

## 2. Evaluation Design

```
run_benchmark.py  →  TTFT, Decode Latency/tok, Static/Dynamic VRAM  (efficiency)
lmms-eval         →  VQAv2, POPE, MME                                (accuracy)
```

| Tool | Samples | Prompt | Purpose |
|------|---------|--------|---------|
| run_benchmark.py | {cfg['num_samples']} | `{{question}} Answer in a complete sentence.` | Forces ≥{cfg.get('min_new_tokens','20')} output tokens for stable Decode Latency |
| lmms-eval | per-task | task-standard post-prompt | Matches ground-truth annotation style |

> The two tools use different prompts intentionally.
> Efficiency and accuracy numbers come from different inference conditions by design.

---

## 3. Efficiency Results

*{cfg['num_samples']} VQAv2 samples · {cfg['num_warmup']} warmup runs · seed={cfg['seed']}*

{_stats_table(summary, static_vram)}

### VRAM Summary

| | Value |
|-|-------|
| Static VRAM (model load) | **{static_vram:.3f} GB** |
| Dynamic VRAM mean (per inference) | **{dyn_s['mean']:.3f} GB** |
| Total peak mean | **{total_vram_mean:.3f} GB** |

> **Static** — determines whether the model fits on the device.  
> **Dynamic** — headroom needed for concurrent processes.

### Charts

#### 3.1 Latency Distributions (Prefill · Decode)
![Latency Distribution](latency_dist_{run_id}.png)

#### 3.2 VRAM Breakdown (Static · Dynamic)
![VRAM Breakdown](vram_breakdown_{run_id}.png)

#### 3.3 Per-sample Latency
![Per-sample Latency](per_sample_latency_{run_id}.png)

{acc_block}

{comp_block}

---

## {section_num}. Methodology

### Speed
- **TTFT** — separate `max_new_tokens=1` call per sample; isolates prefill (visual encoding + prompt).
- **Decode Latency per token** — `(total_latency − TTFT) / output_tokens`.
  `min_new_tokens={cfg.get('min_new_tokens','—')}` ensures ≥20 output tokens so per-token overhead is stable.
- TRT optimises prefill via kernel fusion, decode via KV-cache — reporting them separately attributes gains correctly.
- Timing: `torch.cuda.synchronize()` + `time.perf_counter()`.
- Warmup: {cfg['num_warmup']} runs *(for TRT: verify CUDA-graph capture converges within warmup budget)*.

### Memory
- **Static VRAM** — `torch.cuda.memory_allocated()` after model load, before inference.
- **Dynamic VRAM** — `max_memory_allocated() − static_vram`, reset each sample.

### Accuracy
- VQAv2 `exact_match`: case + punctuation normalised.
- POPE: binary yes/no hallucination probing; F1 primary metric.
- MME: choice-format perception/cognition; insensitive to answer-length distribution.

### Reproducibility
- Efficiency: `random.seed({cfg['seed']})` for VQAv2 subset — identical sample IDs across all backend runs.
- Accuracy: lmms-eval `random_seed=0`, `torch_seed=1234`.
- Docker: `nvcr.io/nvidia/tensorrt-llm/release:0.21.0`

---

*Report generated by `report.py` — Swift-VLM-Flow, CSE 599S, UW*
"""

    tag = f"_{output_tag}" if output_tag else ""
    report_path = f"{config.REPORTS_DIR}/report_{run_id}{tag}.md"
    with open(report_path, "w") as f:
        f.write(md)

    print(f"  latency dist  → {chart_latency}")
    print(f"  vram breakdown→ {chart_vram}")
    print(f"  per-sample    → {chart_per_sample}")
    if chart_compare:
        print(f"  comparison    → {chart_compare}")
    print(f"  report        → {report_path}")
    return report_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Generate VLM benchmark report")
    p.add_argument("--efficiency", type=str, default=None,
                   help="Baseline efficiency JSON (default: latest baseline_*.json)")
    p.add_argument("--trt", type=str, nargs="*", default=None,
                   help="TRT run JSON(s) to compare against baseline")
    p.add_argument("--lmms", type=str, default=None,
                   help="lmms-eval results JSON (optional)")
    p.add_argument("--output_tag", type=str, default="",
                   help="Optional suffix appended to output filename")
    return p.parse_args()


def main():
    args = _parse_args()

    eff_path = args.efficiency or _find_latest(
        os.path.join(config.RAW_DIR, "baseline_*.json")
    )
    eff_paths = [eff_path] + (args.trt or [])

    print("=" * 60)
    print(" report.py — Swift-VLM-Flow")
    for i, p in enumerate(eff_paths):
        print(f"  {'Baseline' if i == 0 else f'TRT #{i}':10}: {p}")
    if args.lmms:
        print(f"  {'lmms-eval':10}: {args.lmms}")
    print("=" * 60)

    report = generate_report(eff_paths, args.lmms, args.output_tag)
    print(f"\nDone → {report}")


if __name__ == "__main__":
    main()
