import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "efficiency"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import spec_metrics


def test_compute_speedup():
    # baseline 20 ms/tok, SD 8 ms/tok -> 2.5x
    assert spec_metrics.compute_speedup(20.0, 8.0) == 2.5
    assert spec_metrics.compute_speedup(20.0, 0.0) == 0.0  # guard divide-by-zero


def test_acceptance_from_counts():
    s = spec_metrics.acceptance_from_counts(accepted_tokens=30, verify_steps=10, max_draft_per_step=6)
    assert s["acceptance_length"] == 3.0           # 30/10
    assert round(s["acceptance_rate"], 3) == 0.5   # 30/(10*6)
    # zero-step guard
    z = spec_metrics.acceptance_from_counts(accepted_tokens=0, verify_steps=0, max_draft_per_step=6)
    assert z["acceptance_length"] == 0.0 and z["acceptance_rate"] == 0.0


def test_summarize_spec_results_adds_acceptance():
    per_sample = [
        {"ttft_ms": 100, "decode_latency_ms_per_tok": 8, "dynamic_vram_gb": 1.0,
         "output_tokens": 50, "acceptance_length": 3.0, "acceptance_rate": 0.5},
        {"ttft_ms": 120, "decode_latency_ms_per_tok": 9, "dynamic_vram_gb": 1.1,
         "output_tokens": 60, "acceptance_length": 2.8, "acceptance_rate": 0.47},
    ]
    out = spec_metrics.summarize_spec_results(per_sample)
    assert "acceptance_length" in out and "acceptance_rate" in out
    assert "decode_latency_ms_per_tok" in out
    assert out["acceptance_length"]["mean"] == 2.9


def test_summarize_handles_missing_acceptance_for_baseline():
    # SD-off baseline rows have no acceptance fields
    per_sample = [
        {"ttft_ms": 100, "decode_latency_ms_per_tok": 18, "dynamic_vram_gb": 1.0, "output_tokens": 50},
        {"ttft_ms": 110, "decode_latency_ms_per_tok": 19, "dynamic_vram_gb": 1.0, "output_tokens": 55},
    ]
    out = spec_metrics.summarize_spec_results(per_sample)
    assert out["decode_latency_ms_per_tok"]["mean"] == 18.5
    assert out["acceptance_length"]["mean"] == 0  # absent -> aggregate of [] -> 0
