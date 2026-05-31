"""spec_metrics.py — speculative-decoding metric helpers.

Reuses efficiency/metrics.aggregate (DRY). Callers must put the efficiency/ dir
on sys.path before importing this module.
"""
import metrics  # from edge-deployment/benchmark/efficiency/


def compute_speedup(baseline_ms_per_tok: float, sd_ms_per_tok: float) -> float:
    """Decode speedup = baseline / sd. Returns 0.0 if sd is non-positive."""
    if sd_ms_per_tok <= 0:
        return 0.0
    return round(baseline_ms_per_tok / sd_ms_per_tok, 3)


def acceptance_from_counts(accepted_tokens: int, verify_steps: int,
                           max_draft_per_step: int) -> dict:
    """Derive EAGLE acceptance metrics from raw counts.

    acceptance_length = accepted_tokens / verify_steps
        (avg tokens accepted per verification step; ~1.0 means little speedup)
    acceptance_rate   = accepted_tokens / (verify_steps * max_draft_per_step)
        (fraction of proposed draft tokens that were accepted)
    """
    length = accepted_tokens / verify_steps if verify_steps > 0 else 0.0
    rate = (accepted_tokens / (verify_steps * max_draft_per_step)
            if verify_steps > 0 and max_draft_per_step > 0 else 0.0)
    return {"acceptance_length": round(length, 3), "acceptance_rate": round(rate, 3)}


def summarize_spec_results(per_sample: list[dict]) -> dict:
    """Aggregate per-sample metrics, including the SD-specific acceptance_* keys.

    Mirrors efficiency/metrics.summarize_results but adds acceptance_length and
    acceptance_rate. Keys absent on a sample (e.g. acceptance on an SD-off
    baseline run) simply aggregate over the empty list -> all-zero stats.
    """
    keys = ["ttft_ms", "decode_latency_ms_per_tok", "dynamic_vram_gb",
            "output_tokens", "acceptance_length", "acceptance_rate"]
    summary = {}
    for k in keys:
        vals = [s[k] for s in per_sample if k in s and s[k] is not None]
        summary[k] = metrics.aggregate(vals)
    return summary
