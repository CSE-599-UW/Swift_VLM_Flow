#!/usr/bin/env python3
"""
report.py  –  VLM Quantization Benchmark Report Generator
Usage:
    python report.py <json1> [json2 ...] --output-dir ./output

Accepts any mix of efficiency JSONs (with 'summary'/'per_sample' keys)
and accuracy JSONs (with 'results' key containing vqa/pope/mme).
Generates:
    output/img/         – all figures
    output/report.md    – markdown report with embedded images
"""

import json
import os
import re
import argparse
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ────────────────────────────── global config ──────────────────────────────
OUTPUT_DIR  = Path("../results/reports")
RESULTS_DIR = Path("../results")

_TS_RE = re.compile(r'(\d{8}_\d{6})')


def _tier_prefix(stem: str) -> str:
    """
    Strip the embedded timestamp and trailing version tag from a filename stem
    to get a stable tier key used for grouping.

    Examples:
      "bf16_20260531_225452"          -> "bf16"
      "int4_awq_20260601_000351"      -> "int4_awq"
      "20260528_085407_bf16_v1"       -> "bf16"   (baseline efficiency)
      "bf16_20260531_214526_bf16_v2"  -> "bf16"
    """
    m = _TS_RE.search(stem)
    if not m:
        return stem
    before = stem[:m.start()].rstrip("_")
    after  = stem[m.end():].lstrip("_")
    after  = re.sub(r"_v\d+$", "", after)   # strip trailing _v1 / _v2
    after  = re.sub(r"^v\d+_?", "", after)  # strip leading  v1_ / v2_
    return before or after or "baseline"


def discover_latest_jsons(results_dir: Path) -> list[Path]:
    """
    Scan the standard four subdirs under *results_dir* and return the newest
    JSON for each (subdir, tier-prefix) pair, sorted by path.
    """
    subdirs = [
        "efficiency/baseline",
        "efficiency/trt",
        "accuracy/baseline",
        "accuracy/trt",
    ]
    best: dict[tuple, tuple] = {}   # (subdir, tier) -> (timestamp_str, Path)
    for sub in subdirs:
        d = results_dir / sub
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            m = _TS_RE.search(f.stem)
            ts     = m.group(1) if m else ""
            prefix = _tier_prefix(f.stem)
            key    = (sub, prefix)
            if key not in best or ts > best[key][0]:
                best[key] = (ts, f)

    paths = sorted(v[1] for v in best.values())
    return paths

# 6 tiers × (efficiency + accuracy) = 12 JSONs for a full official run
INPUT_JSONS = [
    "../results/efficiency/baseline/20260528_085407_bf16_v1.json",
    # "../results/efficiency/trt/bf16_20260528_085459_bf16_v1.json",
    "../results/efficiency/trt/bf16_20260531_214526_bf16_v2.json",
    # "../results/efficiency/trt/fp8_20260528_085545_fp8_v1.json",
    "../results/efficiency/trt/fp8_20260531_214912_fp8_v2.json",
    # "../results/efficiency/trt/int8_20260528_085621_int8_v1.json",
    "../results/efficiency/trt/int8_20260531_214641_int8_v2.json",
    # "../results/efficiency/trt/int4_20260528_085659_int4_v1.json",
    "../results/efficiency/trt/int4_20260531_214737_int4_v2.json",
    # "../results/efficiency/trt/int4_20260529_033633_int4_awq.json",
    "../results/efficiency/trt/int4_awq_20260531_214958_int4_awq_v2.json",
    "../results/efficiency/trt/nvfp4_20260531_215044_nvfp4_v2.json",
    "../results/efficiency/trt/smoothquant_20260531_214820_smoothquant_v2.json",
    "../results/accuracy/baseline/bf16_20260528_042224.json",
    # "../results/accuracy/trt/bf16_20260528_044749.json",
    "../results/accuracy/trt/bf16_20260531_225452.json",
    # "../results/accuracy/trt/fp8_20260529_043513.json",
    "../results/accuracy/trt/fp8_20260601_012129.json",
    # "../results/accuracy/trt/int8_20260528_181038.json",
    "../results/accuracy/trt/int8_20260531_233156.json",
    # "../results/accuracy/trt/int4_20260528_235602.json",
    "../results/accuracy/trt/int4_20260531_234747.json",
    # "../results/accuracy/trt/int4_20260528_063759.json", # awq
    "../results/accuracy/trt/int4_awq_20260601_000351.json",
    # "../results/accuracy/trt/smoothquant_<run_id>.json",
    # "../results/accuracy/trt/nvfp4_<run_id>.json",
    "../results/accuracy/trt/nvfp4_20260601_013737.json",
]

# ────────────────────────────── colour scheme ──────────────────────────────
# Canonical display order
TIER_ORDER = [
    "pytorch-bf16",
    "trt-bf16",
    "trt-int8",
    "trt-int4",
    "trt-smoothquant",
    "trt-fp8",
    "trt-int_awq",
    "trt-nvfp4",
]

TIER_COLORS = {
    "pytorch-bf16":    "#5F5E5A",   # gray     – baseline
    "trt-bf16":        "#185FA5",   # blue
    "trt-int8":        "#BA7517",   # amber
    "trt-int4":        "#993C1D",   # coral
    "trt-smoothquant": "#A83060",   # rose
    "trt-fp8":         "#0F6E56",   # teal
    "trt-int_awq":     "#533AB7",   # purple
    "trt-nvfp4":       "#1A6B8A",   # dark cyan
}

TIER_LABELS = {
    "pytorch-bf16":    "PyTorch BF16",
    "trt-bf16":        "TRT BF16",
    "trt-int8":        "TRT INT8",
    "trt-int4":        "TRT INT4",
    "trt-smoothquant": "TRT SmoothQuant",
    "trt-fp8":         "TRT FP8",
    "trt-int_awq":     "TRT INT4-AWQ",
    "trt-nvfp4":       "TRT NVFP4",
}

# ────────────────────────────── helpers ────────────────────────────────────
def tier_key(config: dict) -> str:
    """Derive a canonical tier string from a run config."""
    backend = config.get("backend", "").lower()
    precision = config.get("precision", "").lower()
    engine_dir = config.get("engine_dir", "").lower()

    if "pytorch" in backend or "huggingface" in backend:
        prec = precision if precision else "bf16"
        return f"pytorch-{prec}"
    if "tensorrt" in backend or "trt" in backend:
        # backend might be  'tensorrt-llm-fp8'  or precision field
        prec = precision if precision else ""
        if not prec:
            for p in ["bf16", "fp8", "int8", "int4", "int_awq", "smoothquant", "nvfp4", "fp16"]:
                if p in backend:
                    prec = p
                    break
        if not prec:
            prec = "bf16"
        # Normalise precision strings to canonical tier names
        if prec == "int4_awq":
            prec = "int_awq"
        # AWQ engine without explicit precision: detect via engine_dir path
        if prec == "int4" and "awq" in engine_dir:
            prec = "int_awq"
        return f"trt-{prec}"
    # fallback
    return f"{backend}-{precision}" if precision else backend


def sorted_tiers(tiers):
    """Return tiers in canonical display order, unknowns appended."""
    known = [t for t in TIER_ORDER if t in tiers]
    unknown = sorted([t for t in tiers if t not in TIER_ORDER])
    return known + unknown


def tier_label(t):
    return TIER_LABELS.get(t, t)


def tier_color(t):
    return TIER_COLORS.get(t, "#888780")


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ────────────────────────────── data loading ───────────────────────────────
def load_runs(json_paths):
    """
    Returns two dicts keyed by tier string:
        eff_runs[tier]  = efficiency summary dict  (ttft, decode, vram, …)
        acc_runs[tier]  = accuracy scores dict     (vqa, pope, mme)
    """
    eff_runs = {}
    acc_runs = {}

    for path in json_paths:
        with open(path) as f:
            d = json.load(f)

        cfg = d.get("config", {})
        # Determine type
        if "summary" in d:          # efficiency JSON
            tier = tier_key(cfg)
            output_tokens_mean = d["summary"].get("output_tokens", {}).get("mean")
            eff_runs[tier] = {
                "ttft_mean":           d["summary"]["ttft_ms"]["mean"],
                "ttft_std":            d["summary"]["ttft_ms"]["std"],
                "ttft_p95":            d["summary"]["ttft_ms"]["p95"],
                "decode_mean":         d["summary"]["decode_latency_ms_per_tok"]["mean"],
                "decode_std":          d["summary"]["decode_latency_ms_per_tok"]["std"],
                "decode_p95":          d["summary"]["decode_latency_ms_per_tok"]["p95"],
                "output_tokens_mean":  output_tokens_mean,
                "static_vram_gb":      d.get("static_vram_gb", None),
                "dynamic_mean_gb":     d["summary"]["dynamic_vram_gb"]["mean"],
                "dynamic_std_gb":      d["summary"]["dynamic_vram_gb"]["std"],
                "config":              cfg,
            }
        elif "results" in d:        # accuracy JSON
            # accuracy JSONs may have backend at top level, not in config
            merged_cfg = {**cfg}
            if "backend" in d and "backend" not in merged_cfg:
                merged_cfg["backend"] = d["backend"]
            tier = tier_key(merged_cfg)
            r = d["results"]
            vqa   = r.get("vqa",  {}).get("scores", {})
            # pope  = r.get("pope", {})
            pope = r.get("pope", {}).get("scores", r.get("pope", {}))
            mme   = r.get("mme",  {}).get("scores", {})
            acc_runs[tier] = {
                "vqa_acc":          vqa.get("accuracy"),
                "pope_avg_f1":      pope.get("avg_f1"),
                "pope_avg_acc":     pope.get("avg_accuracy"),
                "mme_total":        mme.get("total_score"),
                "mme_perception":   mme.get("perception_score"),
                "mme_cognition":    mme.get("cognition_score"),
                "mme_per_task":     mme.get("per_task", {}),
                "config":           cfg,
            }

    return eff_runs, acc_runs


# ────────────────────────── figure generators ──────────────────────────────
def fig_speed(eff_runs, img_dir):
    """Grouped bar: TTFT + Decode Latency side by side."""
    tiers = sorted_tiers(eff_runs.keys())
    if not tiers:
        return None

    x = np.arange(len(tiers))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.35)

    for ax, metric, ylabel, title in [
        (axes[0], "ttft_mean",   "ms",             "TTFT (Prefill Latency)"),
        (axes[1], "decode_mean", "ms / token",      "Decode Latency per Token"),
    ]:
        std_key = metric.replace("mean", "std")
        vals = [eff_runs[t][metric] for t in tiers]
        errs = [eff_runs[t][std_key] for t in tiers]
        colors = [tier_color(t) for t in tiers]

        bars = ax.bar(x, vals, width=0.6, color=colors, yerr=errs,
                      error_kw={"ecolor": "#3d3d3a", "capsize": 4, "linewidth": 1},
                      zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([tier_label(t) for t in tiers], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        # value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errs) * 0.05,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    path = img_dir / "fig_speed.png"
    savefig(fig, path)
    return path


def fig_speedup(eff_runs, img_dir):
    """Normalized speedup bar (baseline = pytorch-bf16)."""
    tiers = sorted_tiers(eff_runs.keys())
    baseline = "pytorch-bf16"
    if baseline not in eff_runs or len(tiers) < 2:
        return None

    b_ttft   = eff_runs[baseline]["ttft_mean"]
    b_decode = eff_runs[baseline]["decode_mean"]
    x = np.arange(len(tiers))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.35)

    for ax, b_val, metric, title in [
        (axes[0], b_ttft,   "ttft_mean",   "TTFT Speedup vs PyTorch BF16"),
        (axes[1], b_decode, "decode_mean", "Decode Speedup vs PyTorch BF16"),
    ]:
        ratios = [b_val / eff_runs[t][metric] for t in tiers]
        colors = [tier_color(t) for t in tiers]

        bars = ax.bar(x, ratios, width=0.6, color=colors, zorder=3)
        ax.axhline(1.0, color="#3d3d3a", linewidth=1, linestyle="--", label="Baseline (1×)")
        ax.set_xticks(x)
        ax.set_xticklabels([tier_label(t) for t in tiers], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Speedup (×)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        for bar, v in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}×", ha="center", va="bottom", fontsize=8)

    path = img_dir / "fig_speedup.png"
    savefig(fig, path)
    return path


def fig_vram(eff_runs, img_dir):
    """Stacked bar: Static + Dynamic VRAM."""
    tiers = sorted_tiers(eff_runs.keys())
    tiers_with_static = [t for t in tiers if eff_runs[t]["static_vram_gb"] is not None]
    if not tiers_with_static:
        return None

    static = [eff_runs[t]["static_vram_gb"] for t in tiers_with_static]
    dynamic = [eff_runs[t]["dynamic_mean_gb"] for t in tiers_with_static]
    x = np.arange(len(tiers_with_static))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.bar(x, static,  width=0.6, label="Static VRAM",
           color=[tier_color(t) for t in tiers_with_static], zorder=3)
    ax.bar(x, dynamic, width=0.6, bottom=static, label="Dynamic VRAM",
           color=[tier_color(t) for t in tiers_with_static],
           alpha=0.45, hatch="///", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([tier_label(t) for t in tiers_with_static], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("VRAM (GB)", fontsize=10)
    ax.set_title("Static + Dynamic VRAM Usage", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # labels on stacked bars
    for i, (s, d) in enumerate(zip(static, dynamic)):
        ax.text(x[i], s / 2, f"{s:.2f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(x[i], s + d / 2, f"+{d:.2f}", ha="center", va="center", fontsize=7.5)

    handles = [
        mpatches.Patch(color="#888780", label="Static VRAM"),
        mpatches.Patch(color="#888780", alpha=0.45, hatch="///", label="Dynamic VRAM"),
    ]
    ax.legend(handles=handles, fontsize=9)

    path = img_dir / "fig_vram.png"
    savefig(fig, path)
    return path


def fig_accuracy(acc_runs, img_dir):
    """Grouped bar: VQA / POPE F1 / MME (normalized to baseline)."""
    tiers = sorted_tiers(acc_runs.keys())
    if not tiers:
        return None

    metrics = [
        ("vqa_acc",      "VQAv2 Accuracy (%)"),
        ("pope_avg_f1",  "POPE Avg F1 (%)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(wspace=0.35)

    for ax, (key, title) in zip(axes, metrics):
        vals = [acc_runs[t].get(key) for t in tiers]
        # skip if all None
        if all(v is None for v in vals):
            ax.set_visible(False)
            continue

        colors = [tier_color(t) for t in tiers]
        x = np.arange(len(tiers))
        bars = ax.bar(x, [v if v is not None else 0 for v in vals],
                      width=0.6, color=colors, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([tier_label(t) for t in tiers], rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("%", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ymin = min(v for v in vals if v is not None) * 0.97
        ax.set_ylim(ymin, 100)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    path = img_dir / "fig_accuracy.png"
    savefig(fig, path)
    return path


def fig_mme(acc_runs, img_dir):
    """MME bar: perception + cognition stacked."""
    tiers = sorted_tiers(acc_runs.keys())
    tiers_with_mme = [t for t in tiers if acc_runs[t].get("mme_total") is not None]
    if not tiers_with_mme:
        return None

    perception = [acc_runs[t]["mme_perception"] for t in tiers_with_mme]
    cognition  = [acc_runs[t]["mme_cognition"]  for t in tiers_with_mme]
    x = np.arange(len(tiers_with_mme))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, perception, width=0.6, label="Perception",
           color=[tier_color(t) for t in tiers_with_mme], zorder=3)
    ax.bar(x, cognition,  width=0.6, bottom=perception, label="Cognition",
           color=[tier_color(t) for t in tiers_with_mme],
           alpha=0.5, hatch="///", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([tier_label(t) for t in tiers_with_mme], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("MME Score", fontsize=10)
    ax.set_title("MME Score (Perception + Cognition)", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    for i, (p, c) in enumerate(zip(perception, cognition)):
        ax.text(x[i], p / 2, f"{p:.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(x[i], p + c / 2, f"+{c:.0f}", ha="center", va="center", fontsize=7.5)

    handles = [
        mpatches.Patch(color="#888780", label="Perception"),
        mpatches.Patch(color="#888780", alpha=0.5, hatch="///", label="Cognition"),
    ]
    ax.legend(handles=handles, fontsize=9)

    path = img_dir / "fig_mme.png"
    savefig(fig, path)
    return path


def fig_tradeoff(eff_runs, acc_runs, img_dir):
    """3-panel scatter: total latency (TTFT + decode) vs VQAv2 / POPE F1 / MME."""
    common = sorted_tiers(set(eff_runs) & set(acc_runs))
    if len(common) < 2:
        return None

    panels = [
        ("vqa_acc",     "VQAv2 Accuracy (%)"),
        ("pope_avg_f1", "POPE Avg F1 (%)"),
        ("mme_total",   "MME Total Score"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.38)

    for ax, (acc_key, ylabel) in zip(axes, panels):
        tiers = [t for t in common
                 if acc_runs[t].get(acc_key) is not None
                 and eff_runs[t].get("ttft_mean") is not None]
        if len(tiers) < 2:
            ax.set_visible(False)
            continue

        for t in tiers:
            e = eff_runs[t]
            n_tok = e.get("output_tokens_mean") or 50
            total_lat = e["ttft_mean"] + e["decode_mean"] * n_tok
            y_val = acc_runs[t][acc_key]
            s_val = (e.get("static_vram_gb") or 6.0) * 80

            ax.scatter(total_lat, y_val, s=s_val, color=tier_color(t),
                       alpha=0.85, edgecolors="#3d3d3a", linewidths=0.6, zorder=3)
            ax.annotate(tier_label(t), (total_lat, y_val),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

        ax.set_xlabel("Total Latency (ms)\n= TTFT + Decode × Output Tokens", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.grid(linestyle="--", alpha=0.4)

    path = img_dir / "fig_tradeoff.png"
    savefig(fig, path)
    return path


# ───────────────────────────── markdown table ──────────────────────────────
def build_table(eff_runs, acc_runs):
    all_tiers = sorted_tiers(set(eff_runs) | set(acc_runs))
    baseline_eff = eff_runs.get("pytorch-bf16")

    header = (
        "| Tier | TTFT (ms) | Decode (ms/tok) | "
        "Static VRAM (GB) | Dyn VRAM (GB) | "
        "VQAv2 (%) | POPE F1 (%) | MME Total |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for t in all_tiers:
        e = eff_runs.get(t, {})
        a = acc_runs.get(t, {})

        def fmt_speed(val, std, baseline_val):
            if val is None:
                return "–"
            s = f"{val:.1f} ±{std:.1f}"
            if baseline_val and t != "pytorch-bf16":
                ratio = baseline_val / val
                s += f" ({ratio:.2f}×)"
            return s

        ttft   = fmt_speed(e.get("ttft_mean"),   e.get("ttft_std"),   baseline_eff.get("ttft_mean")   if baseline_eff else None)
        decode = fmt_speed(e.get("decode_mean"), e.get("decode_std"), baseline_eff.get("decode_mean") if baseline_eff else None)
        static = f"{e['static_vram_gb']:.2f}" if e.get("static_vram_gb") is not None else "–"
        dynamic = f"{e['dynamic_mean_gb']:.3f} ±{e['dynamic_std_gb']:.3f}" if e.get("dynamic_mean_gb") is not None else "–"
        vqa  = f"{a['vqa_acc']:.1f}"   if a.get("vqa_acc")     is not None else "–"
        pope = f"{a['pope_avg_f1']:.1f}" if a.get("pope_avg_f1") is not None else "–"
        mme  = f"{a['mme_total']:.0f}" if a.get("mme_total")   is not None else "–"

        rows.append(f"| **{tier_label(t)}** | {ttft} | {decode} | {static} | {dynamic} | {vqa} | {pope} | {mme} |")

    return "\n".join(rows)


# ───────────────────────────── report builder ──────────────────────────────
def build_report(eff_runs, acc_runs, img_dir, output_dir):
    imgs = {}

    print("Generating figures...")
    if eff_runs:
        p = fig_speed(eff_runs, img_dir)
        if p: imgs["speed"] = p
        p = fig_speedup(eff_runs, img_dir)
        if p: imgs["speedup"] = p
        p = fig_vram(eff_runs, img_dir)
        if p: imgs["vram"] = p

    if acc_runs:
        p = fig_accuracy(acc_runs, img_dir)
        if p: imgs["accuracy"] = p
        p = fig_mme(acc_runs, img_dir)
        if p: imgs["mme"] = p

    if eff_runs and acc_runs:
        p = fig_tradeoff(eff_runs, acc_runs, img_dir)
        if p: imgs["tradeoff"] = p

    def img_link(key, alt):
        if key in imgs:
            rel = os.path.relpath(imgs[key], output_dir)
            return f"![{alt}]({rel})"
        return ""

    # ── assemble tiers metadata ──
    all_tiers = sorted_tiers(set(eff_runs) | set(acc_runs))
    model = None
    for src in [eff_runs, acc_runs]:
        for t in all_tiers:
            if t in src and src[t].get("config", {}).get("model"):
                model = src[t]["config"]["model"]
                break
        if model:
            break
    model_str = model or "Qwen2-VL"

    lines = []
    lines.append(f"# {model_str} — Quantization Benchmark Results\n")
    lines.append("> Auto-generated by `report.py`. Efficiency: mean ± std over benchmark samples. "
                 "Speedup ratio in parentheses = PyTorch BF16 latency ÷ current latency.\n")

    # ── Summary table ──
    lines.append("## Results Summary\n")
    lines.append(build_table(eff_runs, acc_runs))
    lines.append("")

    # ── Speed section ──
    if "speed" in imgs:
        lines.append("## Speed\n")
        lines.append("Prefill (TTFT) and decode latency measured separately. "
                     "Error bars = ±1 std across samples.\n")
        lines.append(img_link("speed", "TTFT and Decode Latency"))
        lines.append("")

    if "speedup" in imgs:
        lines.append("### Speedup vs PyTorch BF16\n")
        lines.append(img_link("speedup", "Speedup"))
        lines.append("")

    # ── Memory section ──
    if "vram" in imgs:
        lines.append("## Memory\n")
        lines.append("Static VRAM = model loaded, before inference. "
                     "Dynamic VRAM = additional peak during one forward pass.\n"
                     "TRT static VRAM include pre-allocated buffer for activation and profile. Therefore, the TRT static VRAM bigger than pytorch baseline.\n")
        lines.append(img_link("vram", "VRAM Usage"))
        lines.append("")

    # ── Accuracy section ──
    if "accuracy" in imgs or "mme" in imgs:
        lines.append("## Accuracy\n")
        lines.append("VQAv2 (500 samples), POPE (adversarial/popular/random subsets), "
                     "MME (full benchmark).\n")
        if "accuracy" in imgs:
            lines.append(img_link("accuracy", "VQA and POPE Accuracy"))
            lines.append("")
        if "mme" in imgs:
            lines.append("### MME Score\n")
            lines.append(img_link("mme", "MME Score"))
            lines.append("")

    # ── Tradeoff ──
    if "tradeoff" in imgs:
        lines.append("## Accuracy–Latency Tradeoff\n")
        lines.append("X-axis: total end-to-end latency (TTFT + decode latency × mean output tokens). "
                     "Each point is one quantization tier. Bubble size scales with static VRAM.\n")
        lines.append(img_link("tradeoff", "Accuracy–Latency Tradeoff"))
        lines.append("")

    # ── Per-task MME detail ──
    mme_detail_tiers = [t for t in all_tiers
                        if t in acc_runs and acc_runs[t].get("mme_per_task")]
    if mme_detail_tiers:
        # Collect union of all task names across tiers, sorted
        all_tasks = sorted({
            task
            for t in mme_detail_tiers
            for task in acc_runs[t]["mme_per_task"]
        })
        # Header
        tier_hdrs = " | ".join(f"**{tier_label(t)}**" for t in mme_detail_tiers)
        lines.append("## MME Per-Task Detail\n")
        lines.append(f"| Task | {tier_hdrs} |")
        lines.append("|" + "---|" * (1 + len(mme_detail_tiers)))
        for task in all_tasks:
            cells = []
            for t in mme_detail_tiers:
                td = acc_runs[t]["mme_per_task"].get(task)
                if td:
                    cells.append(f"{td['score']:.0f}/{td['max_score']:.0f} ({td['accuracy']:.1f}%)")
                else:
                    cells.append("–")
            lines.append(f"| {task} | " + " | ".join(cells) + " |")
        lines.append("")

    # ── POPE detail ──
    pope_detail_lines = []
    for t in all_tiers:
        if t not in acc_runs:
            continue
        a = acc_runs[t]
        if a.get("pope_avg_f1") is not None:
            pope_detail_lines.append(
                f"- **{tier_label(t)}**: avg F1 = {a['pope_avg_f1']:.1f}%, avg acc = {a.get('pope_avg_acc', '–')}"
            )
    if pope_detail_lines:
        lines.append("## POPE Detail\n")
        lines.append("\n".join(pope_detail_lines))
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ────────────────────────────────── main ───────────────────────────────────
def create_parser():
    parser = argparse.ArgumentParser(description="Generate benchmark report from JSON files")
    parser.add_argument("jsons", nargs="*",
                        help="Efficiency or accuracy JSON files (overrides INPUT_JSONS if provided)")
    parser.add_argument("--output-dir", default=None,
                        help="Root output directory (overrides OUTPUT_DIR if provided)")
    parser.add_argument("--latest", action="store_true",
                        help="Auto-discover the newest JSON per tier under --results-dir")
    parser.add_argument("--results-dir", default=None,
                        help="Root of the results tree used by --latest "
                             "(default: ../results relative to this script)")
    return parser


def main():
    args = create_parser().parse_args()

    if args.latest:
        script_dir   = Path(__file__).resolve().parent
        results_root = Path(args.results_dir).resolve() if args.results_dir \
                       else (script_dir / RESULTS_DIR).resolve()
        json_paths = discover_latest_jsons(results_root)
        print(f"Auto-discovered {len(json_paths)} latest JSON(s) under {results_root}:")
        for p in json_paths:
            print(f"  {p.relative_to(results_root)}")
    elif args.jsons:
        json_paths = args.jsons
    else:
        json_paths = INPUT_JSONS

    base_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir    = base_dir / f"report_{timestamp}"
    img_dir    = run_dir / "img"
    run_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    print(f"Loading {len(json_paths)} JSON file(s)...")
    eff_runs, acc_runs = load_runs(json_paths)

    print(f"  Efficiency runs found: {sorted_tiers(eff_runs.keys())}")
    print(f"  Accuracy runs found:   {sorted_tiers(acc_runs.keys())}")

    report_path = build_report(eff_runs, acc_runs, img_dir, run_dir)
    print(f"\nReport written to: {report_path}")
    print(f"Images saved in:   {img_dir}")


if __name__ == "__main__":
    main()
