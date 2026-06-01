"""run_spec_trt_edge.py — EAGLE3 speculative-decoding benchmark runner (TensorRT-Edge-LLM).

Drives the C++ `llm_inference` binary as a subprocess over LLaVA-Bench images, in
two modes:
  --spec_decode      : SD-on  (EAGLE base+draft engines, engines/llm)
  --no-spec_decode   : SD-off (vanilla base engine,      engines/llm_vanilla)

Emits JSON in Kevin's efficiency schema extended with SD fields (acceptance_length,
acceptance_rate, spec_decode, draft_model, speedup_vs_base) to results/speculative/.

Prerequisite: source edge-deployment/benchmark/speculative/env.sh first
(sets CUDA_HOME, TRT_PACKAGE_DIR, LD_LIBRARY_PATH, EDGE_LLM_PATH, WORKSPACE_DIR).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "efficiency"))
sys.path.insert(0, os.path.dirname(__file__))

import config_spec
import io_builder
import output_parser
import spec_metrics
import data_loader  # from efficiency/


def parse_args():
    ap = argparse.ArgumentParser(description="Qwen2.5-VL EAGLE3 SD benchmark (TRT-Edge-LLM)")
    ap.add_argument("--num_samples", type=int, default=config_spec.NUM_SAMPLES)
    ap.add_argument("--warmup", type=int, default=config_spec.NUM_WARMUP)
    ap.add_argument("--max_new_tokens", type=int, default=config_spec.MAX_NEW_TOKENS)
    ap.add_argument("--spec_decode", action=argparse.BooleanOptionalAction, default=True,
                    help="--spec_decode (EAGLE) or --no-spec_decode (vanilla baseline)")
    ap.add_argument("--baseline_via_eagle", action="store_true",
                    help="for the SD-off baseline, decode on the EAGLE engine with per-request "
                         "disable_spec_decode (same-engine baseline) instead of a separate vanilla "
                         "engine — use when no vanilla engine exists (e.g. fp8)")
    ap.add_argument("--output_tag", type=str, default=None)
    return ap.parse_args()


def save_images(samples: list[dict], dst_dir: str) -> list[dict]:
    """Persist each PIL image to disk; return harness samples with absolute image paths."""
    os.makedirs(dst_dir, exist_ok=True)
    out = []
    for s in samples:
        qid = s["question_id"]
        path = os.path.abspath(os.path.join(dst_dir, f"img_{qid}.png"))
        s["image"].save(path)
        out.append({"question_id": qid, "prompt": s["question"], "image_path": path,
                    "answers": s.get("answers", [])})
    return out


def run_inference(input_path: str, output_path: str, profile_path: str,
                  spec_decode: bool, warmup: int, max_new_tokens: int,
                  baseline_via_eagle: bool = False) -> None:
    # The EAGLE engine is used for SD-on and for the same-engine SD-off baseline
    # (baseline_via_eagle, where per-request disable_spec_decode turns speculation off).
    use_eagle_engine = spec_decode or baseline_via_eagle
    engine_dir = config_spec.ENGINE_LLM_DIR if use_eagle_engine else config_spec.ENGINE_LLM_VANILLA_DIR
    cmd = [
        config_spec.LLM_INFERENCE_BIN,
        "--engineDir", engine_dir,
        "--multimodalEngineDir", config_spec.ENGINE_VISUAL_DIR,
        "--inputFile", input_path,
        "--outputFile", output_path,
        "--profileOutputFile", profile_path,
        "--dumpProfile",
        "--warmup", str(warmup),
        "--maxGenerateLength", str(max_new_tokens),
    ]
    # --specDecode initializes the spec runtime / loads the draft. For the
    # baseline_via_eagle case it's still passed; per-request disable_spec_decode
    # (set in the input.json) then forces autoregressive base-only decoding.
    if use_eagle_engine:
        cmd += ["--specDecode",
                "--specDraftTopK", str(config_spec.SPEC_DRAFT_TOPK),
                "--specDraftStep", str(config_spec.SPEC_DRAFT_STEP),
                "--specVerifyTreeSize", str(config_spec.SPEC_VERIFY_TREE)]
    # Run with cwd=EDGE_LLM_PATH so the default plugin path (build/lib...) resolves.
    log = subprocess.run(cmd, cwd=config_spec.EDGE_LLM_PATH,
                         capture_output=True, text=True)
    if log.returncode != 0:
        sys.stderr.write(log.stdout[-4000:] + "\n" + log.stderr[-4000:] + "\n")
        raise RuntimeError(f"llm_inference failed (rc={log.returncode})")


def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "spec" if args.spec_decode else "base"
    print("=" * 60)
    print(f" Qwen2.5-VL-7B EAGLE3 SD benchmark — mode={mode} ({config_spec.PRECISION})")
    print(f" backend: trt-edge-llm | samples={args.num_samples} | run_id={run_id}")
    print("=" * 60)

    print("[1/4] Loading LLaVA-Bench samples...")
    samples = data_loader.load_llava_bench_samples(num_samples=args.num_samples)

    work = tempfile.mkdtemp(prefix=f"sdbench_{mode}_")
    hsamples = save_images(samples, os.path.join(work, "imgs"))
    input_path = os.path.join(work, "input.json")
    output_path = os.path.join(work, "output.json")
    profile_path = os.path.join(work, "profile.json")
    base_via_eagle = (not args.spec_decode) and args.baseline_via_eagle
    io_builder.write_input_json(
        io_builder.build_input_json(hsamples, max_new_tokens=args.max_new_tokens,
                                    temperature=0.0, top_k=1, top_p=1.0,
                                    disable_spec_decode=base_via_eagle),
        input_path)

    print(f"[2/4] Running llm_inference (spec_decode={args.spec_decode}"
          f"{', baseline_via_eagle' if base_via_eagle else ''})...")
    run_inference(input_path, output_path, profile_path,
                  args.spec_decode, args.warmup, args.max_new_tokens,
                  baseline_via_eagle=args.baseline_via_eagle)

    print("[3/4] Parsing profile + outputs...")
    prof = output_parser.parse_profile(profile_path, spec_decode=args.spec_decode)
    outs = output_parser.parse_outputs(output_path)

    # per_sample carries predicted text (for the SD-on/off parity check)
    per_sample = []
    for o in outs:
        idx = o["request_idx"]
        qid = hsamples[idx]["question_id"] if idx is not None and idx < len(hsamples) else idx
        per_sample.append({"question_id": qid, "predicted_answer": o["output_text"]})

    summary = {
        "decode_latency_ms_per_tok": prof["decode_latency_ms_per_tok"],
        "decode_tokens_per_sec":     prof["decode_tokens_per_sec"],
        "ttft_ms_prefill_mean":      prof["prefill_ms_mean"],
        "ttft_ms_prefill_p95":       prof["prefill_ms_p95"],
        "vision_ms":                 prof["vision_ms"],
        "acceptance_length":         prof["acceptance_length"],
        "output_tokens":             prof["output_tokens"],
        "verify_steps":              prof["verify_steps"],
    }
    out = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": config_spec.MODEL_NAME,
            "draft_model": config_spec.DRAFT_MODEL if args.spec_decode else None,
            "precision": config_spec.PRECISION,
            "backend": "trt-edge-llm",
            "spec_decode": args.spec_decode,
            "engine_dir": (config_spec.ENGINE_LLM_DIR if (args.spec_decode or base_via_eagle)
                           else config_spec.ENGINE_LLM_VANILLA_DIR),
            "baseline_method": (None if args.spec_decode else
                                ("disable_spec_decode_on_eagle" if base_via_eagle
                                 else "vanilla_engine")),
            "num_samples": args.num_samples,
            "num_warmup": args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "spec_draft_topk": config_spec.SPEC_DRAFT_TOPK if args.spec_decode else None,
            "spec_draft_step": config_spec.SPEC_DRAFT_STEP if args.spec_decode else None,
            "spec_verify_tree": config_spec.SPEC_VERIFY_TREE if args.spec_decode else None,
            "dataset": config_spec.DATASET,
            "seed": config_spec.SAMPLE_SEED,
        },
        "peak_vram_gb": prof["peak_vram_gb"],
        "summary": summary,
        "per_sample": per_sample,
    }

    os.makedirs(config_spec.RESULTS_DIR, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    fn = os.path.join(config_spec.RESULTS_DIR, f"{mode}_{run_id}{tag}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)

    print("[4/4] Done.")
    print(f"  decode: {prof['decode_tokens_per_sec']:.1f} tok/s "
          f"({prof['decode_latency_ms_per_tok']:.2f} ms/tok)")
    if args.spec_decode:
        print(f"  acceptance length: {prof['acceptance_length']:.2f} tokens/step")
    print(f"  prefill mean: {prof['prefill_ms_mean']:.1f} ms | vision: {prof['vision_ms']:.1f} ms")
    print(f"  peak VRAM: {prof['peak_vram_gb']:.2f} GB")
    print(f"  → {fn}")


if __name__ == "__main__":
    main()
