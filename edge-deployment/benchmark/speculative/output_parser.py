"""output_parser.py — parse TensorRT-Edge-LLM llm_inference artifacts.

Reads the structured profile JSON (--profileOutputFile) and the responses JSON
(--outputFile). The profiler aggregates over all requests in one input file, so
these are run-level metrics plus per-request output text.

Returns a flat dict of the metrics the SD benchmark records. Handles both the
EAGLE spec-decode profile (key "eagle_generation") and the vanilla profile
(key "generation").
"""
import json


def _stage(profile: dict, stage_id: str) -> dict | None:
    for s in profile.get("stages", []):
        if s.get("stage_id") == stage_id:
            return s
    return None


def parse_profile(profile_path: str, spec_decode: bool) -> dict:
    """Extract run-level metrics from a profile.json.

    Returns keys:
      decode_tokens_per_sec, decode_latency_ms_per_tok,
      prefill_ms_mean, prefill_ms_p95, vision_ms,
      output_tokens, verify_steps, accepted_tokens,
      acceptance_length, peak_vram_gb
    """
    with open(profile_path) as f:
        p = json.load(f)

    out = {}
    out["peak_vram_gb"] = round(p.get("peak_unified_memory_mb", 0.0) / 1024.0, 3)

    # Prefill (TTFT proxy): distribution across requests from the llm_prefill stage
    prefill_stage = _stage(p, "llm_prefill")
    if prefill_stage:
        gs = prefill_stage["gpu_time_stats"]
        out["prefill_ms_mean"] = round(gs["mean_ms"], 3)
        out["prefill_ms_p95"] = round(gs["p95_ms"], 3)
    else:
        pf = p.get("prefill", {})
        out["prefill_ms_mean"] = round(pf.get("average_time_per_run_ms", 0.0), 3)
        out["prefill_ms_p95"] = out["prefill_ms_mean"]

    vis = _stage(p, "vision_encoder")
    out["vision_ms"] = round(vis["gpu_time_stats"]["mean_ms"], 3) if vis else 0.0

    if spec_decode and "eagle_generation" in p:
        eg = p["eagle_generation"]
        out["output_tokens"] = int(eg.get("total_generated_tokens", 0))
        out["verify_steps"] = int(eg.get("total_iterations", 0))
        out["acceptance_length"] = round(eg.get("average_acceptance_rate", 0.0), 3)
        out["accepted_tokens"] = out["output_tokens"]  # accepted == generated in EAGLE
        tps = eg.get("overall_tokens_per_second_excluding_base_prefill", 0.0)
    else:
        out["verify_steps"] = 0
        out["accepted_tokens"] = 0
        out["acceptance_length"] = 0.0
        # (a) vanilla generation section (separate vanilla engine): key "generation"
        #     with "generated_tokens" — confirmed against a real llm_vanilla profile.
        gen = p.get("generation", p.get("decode", {}))
        if gen:
            out["output_tokens"] = int(gen.get("generated_tokens",
                                                gen.get("total_generated_tokens", 0)))
            tps = gen.get("tokens_per_second",
                          gen.get("overall_tokens_per_second_excluding_base_prefill", 0.0))
        else:
            # (b) base-only via disable_spec_decode on an EAGLE engine: no top-level
            #     generation section. Decode runs one base forward per token, timed in
            #     the llm_generation stage → tok/s = 1000 / mean_ms_per_run.
            st = _stage(p, "llm_generation")
            if st:
                mean_ms = st["gpu_time_stats"]["mean_ms"]
                out["output_tokens"] = int(st.get("total_runs",
                                                   st["gpu_time_stats"].get("count", 0)))
                tps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
            else:
                out["output_tokens"] = 0
                tps = 0.0

    out["decode_tokens_per_sec"] = round(tps, 3)
    out["decode_latency_ms_per_tok"] = round(1000.0 / tps, 3) if tps > 0 else 0.0
    return out


def parse_outputs(output_path: str) -> list[dict]:
    """Return [{request_idx, output_text}] from an llm_inference responses file."""
    with open(output_path) as f:
        d = json.load(f)
    res = []
    for r in d.get("responses", []):
        res.append({
            "request_idx": r.get("request_idx", r.get("batch_idx")),
            "output_text": r.get("output_text", ""),
        })
    return res
