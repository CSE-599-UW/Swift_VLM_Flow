import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "efficiency"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import report_spec


def _mk(spec_decode, ms_per_tok, tok_s, accept, ttft, vram):
    return {
        "config": {"spec_decode": spec_decode, "precision": "bf16"},
        "peak_vram_gb": vram,
        "summary": {
            "decode_latency_ms_per_tok": ms_per_tok,
            "decode_tokens_per_sec": tok_s,
            "ttft_ms_prefill_mean": ttft,
            "acceptance_length": accept,
            "output_tokens": 1000,
        },
    }


def test_build_comparison():
    base = _mk(False, 70.0, 14.3, 0.0, 110.0, 13.6)
    spec = _mk(True, 24.0, 41.5, 3.5, 120.0, 13.95)
    row = report_spec.build_comparison(base, spec)
    assert abs(row["speedup"] - (70.0 / 24.0)) < 0.01
    assert row["acceptance_length"] == 3.5
    assert row["base_decode_tok_s"] == 14.3
    assert row["spec_decode_tok_s"] == 41.5
    assert row["base_vram_gb"] == 13.6 and row["spec_vram_gb"] == 13.95


def test_render_markdown_table():
    base = _mk(False, 70.0, 14.3, 0.0, 110.0, 13.6)
    spec = _mk(True, 24.0, 41.5, 3.5, 120.0, 13.95)
    md = report_spec.render_markdown(report_spec.build_comparison(base, spec))
    assert "speedup" in md.lower() and "acceptance" in md.lower()
    assert "2.92" in md  # 70/24 = 2.92x
