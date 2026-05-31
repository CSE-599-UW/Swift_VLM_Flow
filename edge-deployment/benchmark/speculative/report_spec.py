"""report_spec.py — speedup/acceptance report for the EAGLE3 SD benchmark.

Reads the SD-off (base_*.json) and SD-on (spec_*.json) results produced by
run_spec_trt_edge.py, computes the decode speedup, and emits:
  - fig_spec_speedup.png   (decode tok/s: base vs SD + speedup label)
  - fig_spec_acceptance.png (acceptance length + decode ms/tok)
  - report_spec.md         (summary table)

Usage:
  python report_spec.py [base.json spec.json] [--output-dir DIR]
  (with no paths, picks the newest base_*.json and spec_*.json in results/speculative/)

Standalone by design — keeps Kevin's report.py untouched (refinement of the spec's
"extend report.py").
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "efficiency"))
sys.path.insert(0, os.path.dirname(__file__))
import config_spec
import spec_metrics


def build_comparison(base: dict, spec: dict) -> dict:
    b, s = base["summary"], spec["summary"]
    return {
        "precision": spec["config"].get("precision", "?"),
        "speedup": spec_metrics.compute_speedup(b["decode_latency_ms_per_tok"],
                                                s["decode_latency_ms_per_tok"]),
        "acceptance_length": s.get("acceptance_length", 0.0),
        "base_decode_tok_s": b["decode_tokens_per_sec"],
        "spec_decode_tok_s": s["decode_tokens_per_sec"],
        "base_decode_ms_per_tok": b["decode_latency_ms_per_tok"],
        "spec_decode_ms_per_tok": s["decode_latency_ms_per_tok"],
        "base_ttft_ms": b.get("ttft_ms_prefill_mean", 0.0),
        "spec_ttft_ms": s.get("ttft_ms_prefill_mean", 0.0),
        "base_vram_gb": base.get("peak_vram_gb", 0.0),
        "spec_vram_gb": spec.get("peak_vram_gb", 0.0),
        "num_samples": spec["config"].get("num_samples"),
    }


def render_markdown(c: dict) -> str:
    return f"""# Qwen2.5-VL-7B EAGLE3 Speculative Decoding — TensorRT-Edge-LLM (GB10, {c['precision']})

Decode speedup from EAGLE3 speculative decoding vs the same base model decoding
autoregressively. LLaVA-Bench-in-the-Wild, {c['num_samples']} samples, greedy.

| Metric | Base (SD-off) | EAGLE3 (SD-on) | Gain |
|---|---|---|---|
| Decode throughput (tok/s) | {c['base_decode_tok_s']:.1f} | {c['spec_decode_tok_s']:.1f} | **{c['speedup']:.2f}×** |
| Decode latency (ms/tok) | {c['base_decode_ms_per_tok']:.2f} | {c['spec_decode_ms_per_tok']:.2f} | {c['speedup']:.2f}× faster |
| Acceptance length (tokens/step) | — | {c['acceptance_length']:.2f} | — |
| Prefill / TTFT (ms) | {c['base_ttft_ms']:.1f} | {c['spec_ttft_ms']:.1f} | — |
| Peak VRAM (GB) | {c['base_vram_gb']:.2f} | {c['spec_vram_gb']:.2f} | {c['spec_vram_gb']-c['base_vram_gb']:+.2f} |

**Headline: EAGLE3 speculative decoding delivers a {c['speedup']:.2f}× decode speedup**
(acceptance length {c['acceptance_length']:.2f} tokens/verification step).
"""


def render_figures(c: dict, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []

    # Fig 1: decode throughput bar
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Base\n(SD-off)", "EAGLE3\n(SD-on)"],
                  [c["base_decode_tok_s"], c["spec_decode_tok_s"]],
                  color=["#888", "#76b900"])
    ax.set_ylabel("Decode throughput (tokens/s)")
    ax.set_title(f"Qwen2.5-VL-7B {c['precision']} — {c['speedup']:.2f}× decode speedup (GB10)")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.1f}", ha="center", va="bottom")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "fig_spec_speedup.png")
    fig.savefig(p1, dpi=120); plt.close(fig); paths.append(p1)

    # Fig 2: acceptance length + ms/tok
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4))
    a1.bar(["EAGLE3"], [c["acceptance_length"]], color="#76b900")
    a1.set_ylabel("Acceptance length (tokens/step)"); a1.set_title("EAGLE3 acceptance")
    a1.text(0, c["acceptance_length"], f"{c['acceptance_length']:.2f}", ha="center", va="bottom")
    a2.bar(["Base", "EAGLE3"], [c["base_decode_ms_per_tok"], c["spec_decode_ms_per_tok"]],
           color=["#888", "#76b900"])
    a2.set_ylabel("Decode latency (ms/token)"); a2.set_title("Per-token decode latency")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "fig_spec_acceptance.png")
    fig.savefig(p2, dpi=120); plt.close(fig); paths.append(p2)
    return paths


def _newest(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="optional base.json spec.json")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if len(args.paths) == 2:
        base_p, spec_p = args.paths
    else:
        base_p = _newest(os.path.join(config_spec.RESULTS_DIR, "base_*.json"))
        spec_p = _newest(os.path.join(config_spec.RESULTS_DIR, "spec_*.json"))
    if not base_p or not spec_p:
        sys.exit("Need both a base_*.json and spec_*.json (run run_spec_all.sh first).")

    base, spec = json.load(open(base_p)), json.load(open(spec_p))
    comp = build_comparison(base, spec)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "reports", f"report_spec_{ts}")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    md = render_markdown(comp)
    with open(os.path.join(out_dir, "report_spec.md"), "w") as f:
        f.write(md)
    figs = render_figures(comp, out_dir)

    print(md)
    print(f"\nWrote: {os.path.join(out_dir, 'report_spec.md')}")
    for p in figs:
        print(f"       {p}")
    print(f"\n(base={os.path.basename(base_p)}, spec={os.path.basename(spec_p)})")


if __name__ == "__main__":
    main()
