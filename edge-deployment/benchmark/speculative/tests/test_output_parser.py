import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import output_parser

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def test_parse_spec_profile():
    res = output_parser.parse_profile(
        os.path.join(FIX, "sample_profile_spec.json"), spec_decode=True)
    assert res["output_tokens"] == 132
    assert res["verify_steps"] == 35
    assert abs(res["acceptance_length"] - 3.771) < 0.01
    assert res["decode_tokens_per_sec"] > 0
    assert abs(res["decode_latency_ms_per_tok"] - 1000.0 / res["decode_tokens_per_sec"]) < 0.01
    assert res["peak_vram_gb"] > 10  # ~13.6 GB
    assert res["prefill_ms_mean"] > 0
    assert res["vision_ms"] > 0


def test_parse_outputs():
    outs = output_parser.parse_outputs(os.path.join(FIX, "sample_output_spec.json"))
    assert len(outs) == 1
    assert outs[0]["request_idx"] == 0
    assert "beach" in outs[0]["output_text"].lower()


def test_parse_vanilla_profile():
    # SD-off baseline fixture (captured from a vanilla run)
    path = os.path.join(FIX, "sample_profile_base.json")
    if not os.path.exists(path):
        import pytest
        pytest.skip("vanilla fixture not captured yet")
    res = output_parser.parse_profile(path, spec_decode=False)
    assert res["acceptance_length"] == 0.0
    assert res["verify_steps"] == 0
    assert res["decode_tokens_per_sec"] > 0
    assert res["output_tokens"] > 0


def test_parse_base_only_via_disable_spec():
    # SD-off baseline run on the EAGLE engine with disable_spec_decode: there is no
    # top-level "generation" section; decode timing lives in the llm_generation stage.
    res = output_parser.parse_profile(
        os.path.join(FIX, "sample_profile_base_disable_spec.json"), spec_decode=False)
    assert res["acceptance_length"] == 0.0
    assert res["verify_steps"] == 0
    assert res["output_tokens"] == 164                 # 1 base forward per token
    assert abs(res["decode_latency_ms_per_tok"] - 36.617) < 0.1   # = mean_ms of llm_generation
    assert abs(res["decode_tokens_per_sec"] - 1000.0 / 36.617) < 0.1
    assert abs(res["peak_vram_gb"] - 7.485) < 0.05     # ~7.5 GB (fp8)
    assert res["prefill_ms_mean"] > 0
