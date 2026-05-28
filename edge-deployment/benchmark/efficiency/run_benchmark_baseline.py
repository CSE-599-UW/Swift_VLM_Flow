"""
run_benchmark.py
Main entry point for the VLM Baseline Benchmark.

Workflow:
  1. Load Qwen2-VL-2B model (bf16, PyTorch backend)
  2. Load VQAv2 samples via data_loader
  3. Warm up the model
  4. Run inference on each sample, recording per-sample metrics
  5. Compute aggregate statistics via metrics.py
  6. Save raw results JSON and trigger report generation

Usage:
    python3 run_benchmark.py [--num_samples N] [--warmup W]
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import config
import data_loader
import metrics

# ── Argument Parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="VLM Baseline Benchmark")
    parser.add_argument("--num_samples", type=int, default=config.NUM_SAMPLES,
                        help="Number of VQAv2 samples to evaluate")
    parser.add_argument("--warmup", type=int, default=config.NUM_WARMUP,
                        help="Number of warmup runs")
    parser.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS,
                        help="Max tokens to generate per sample")
    parser.add_argument("--output_tag", type=str, default=None,
                        help="Optional tag appended to output filenames")
    return parser.parse_args()


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model():
    metrics.reset_vram_stats()

    t0 = time.perf_counter()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=config.DEVICE,
    )
    processor = AutoProcessor.from_pretrained(config.MODEL_PATH)
    load_time = time.perf_counter() - t0

    static_vram_gb = metrics.measure_static_vram_gb()
    model.eval()

    print(f"  Model         : {config.MODEL_NAME}")
    print(f"  Precision     : {config.PRECISION}")
    print(f"  Load time     : {load_time:.2f}s")
    print(f"  Static VRAM   : {static_vram_gb:.2f} GB")

    return model, processor, static_vram_gb


# ── Single-sample Inference ────────────────────────────────────────────────────

def run_single(model, processor, sample: dict, max_new_tokens: int, static_vram_gb: float) -> dict:
    """
    Run inference on one VQAv2 sample and return per-sample metrics.

    Args:
        static_vram_gb: Static VRAM baseline measured once after model load.
                        Used to compute the per-inference dynamic VRAM increment.
 
    Returns:
        dict with keys: question_id, question, predicted_answer,
                        ground_truth_answers, ttft_ms, total_latency_ms,
                        throughput_tok_per_sec, dynamic_vram_gb, output_tokens.

    # Returns:
    #     dict with keys: question_id, question, predicted_answer,
    #                     ground_truth_answers, ttft_ms, total_latency_ms,
    #                     throughput_tok_per_sec, peak_vram_gb, output_tokens.
    """
    # =================================================================================
    # Part 1: Dataset input preprocess
    # =================================================================================
    # Add prompt template we set in config
    prompt = data_loader.format_prompt(sample["question"])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    # messages -> pure text token
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # messages -> image_inputs tensor
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(config.DEVICE)

    input_len = inputs["input_ids"].shape[1]  # For cut out input from `full_out` later.


    # =================================================================================
    # Part 2: Timer and Measurement
    # =================================================================================

    # ── TTFT measurement via token-by-token generation ─────────────────────────
    # We use greedy decoding one token at a time to capture TTFT precisely.
    # do_sample=False, temperature=None、top_p=None、top_k=None ==> greedy decoding, excluding random sampling
    timer = metrics.LatencyTimer()
    metrics.reset_vram_stats()

    timer.start()

    # Generate first token only → TTFT
    with torch.no_grad():
        first_out = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    timer.mark_first_token()

    # Generate remaining tokens → total latency
    with torch.no_grad():
        full_out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    timer.stop()

    output_tokens = full_out.shape[1] - input_len
    dynamic_vram = metrics.measure_dynamic_vram_gb(static_vram_gb)
    decode_latency_per_tok = metrics.compute_decode_latency_per_token(
        timer.total_latency_ms, timer.ttft_ms, output_tokens
    )

    predicted_ids = full_out[0][input_len:]
    predicted_answer = processor.decode(predicted_ids, skip_special_tokens=True).strip()

    return {
        "question_id":           sample["question_id"],
        "question":              sample["question"],
        "predicted_answer":      predicted_answer,
        "ground_truth_answers":  sample["answers"],
        "ttft_ms":               round(timer.ttft_ms, 3),
        "total_latency_ms":      round(timer.total_latency_ms, 3),
        "decode_latency_ms_per_tok":round(decode_latency_per_tok, 3),
        "dynamic_vram_gb":       round(dynamic_vram, 3),
        "output_tokens":         output_tokens,
    }


# ── Warmup ─────────────────────────────────────────────────────────────────────

def warmup(model, processor, samples: list, num_warmup: int, max_new_tokens: int,
           static_vram_gb: float):
    print(f"\n[Step 3/5] Warming up ({num_warmup} runs)...")
    for i in range(min(num_warmup, len(samples))):
        result = run_single(model, processor, samples[i], max_new_tokens, static_vram_gb)
        print(
            f"  Warmup {i+1}/{num_warmup} | "
            f"Latency={result['total_latency_ms']:.1f}ms"
        )


# ── Main Benchmark Loop ────────────────────────────────────────────────────────

def run_benchmark(model, processor, samples: list, max_new_tokens: int, static_vram_gb: float) -> list[dict]:
    print(f"\n[Step 4/5] Benchmarking {len(samples)} samples...")
    results = []
    errors = 0

    for i, sample in enumerate(samples):
        try:
            result = run_single(model, processor, sample, max_new_tokens, static_vram_gb)
            results.append(result)
            print(
                f"  [{i+1:>3}/{len(samples)}] "
                f"TTFT={result['ttft_ms']:.1f}ms | "
                f"Dec={result['decode_latency_ms_per_tok']:.2f}ms/tok | "
                f"dVRAM={result['dynamic_vram_gb']:.2f}GB"
            )
        except Exception as e:
            errors += 1
            print(f"  [{i+1:>3}/{len(samples)}] ERROR: {e}")
            traceback.print_exc()

    print(f"\n  Completed: {len(results)}/{len(samples)} samples ({errors} errors)")
    return results


# ── Save Results ───────────────────────────────────────────────────────────────

def save_results(per_sample: list, summary: dict, static_vram_gb: float,
                 args, run_id: str) -> str:
    os.makedirs(config.RAW_DIR, exist_ok=True)

    output = {
        "run_id":        run_id,
        "timestamp":     datetime.now().isoformat(),
        "config": {
            "model":          config.MODEL_NAME,
            "precision":      config.PRECISION,
            "backend":        config.BACKEND,
            "num_samples":    args.num_samples,
            "num_warmup":     args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "dataset":        config.VQAV2_DATASET,
            "split":          config.VQAV2_SPLIT,
            "seed":           config.VQAV2_SEED,
        },
        "static_vram_gb": round(static_vram_gb, 3),
        "summary":        summary,
        "per_sample":     per_sample,
    }

    tag = f"_{args.output_tag}" if args.output_tag else ""
    filename = f"{config.RAW_DIR}/baseline/{run_id}{tag}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Raw results saved → {filename}")
    return filename


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print(" VLM Baseline Benchmark")
    print(f" Model    : {config.MODEL_NAME} ({config.PRECISION})")
    print(f" Backend  : {config.BACKEND}")
    print(f" Samples  : {args.num_samples} (VQAv2 {config.VQAV2_SPLIT})")
    print(f" Run ID   : {run_id}")
    print("=" * 65)

    # Step 1: Load model
    print("\n[Step 1/5] Loading model...")
    model, processor, static_vram_gb = load_model()

    # Step 2: Load data
    print(f"\n[Step 2/5] Loading VQAv2 samples...")
    # samples = data_loader.load_vqav2_samples(num_samples=args.num_samples)
    samples = data_loader.load_llava_bench_samples(num_samples=args.num_samples)

    # Step 3: Warmup
    warmup(model, processor, samples, args.warmup, args.max_new_tokens, static_vram_gb)

    # Step 4: Benchmark
    per_sample = run_benchmark(model, processor, samples, args.max_new_tokens, static_vram_gb)

    # Step 5: Summarize + save + report
    print(f"\n[Step 5/5] Generating report...")
    summary = metrics.summarize_results(per_sample)

    raw_path = save_results(per_sample, summary, static_vram_gb, args, run_id)

    # Print summary to terminal
    print("\n" + "=" * 65)
    print(" RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'p95':>10}")
    print(f"  {'-'*60}")
    for metric, stats in summary.items():
        print(f"  {metric:<30} {stats['mean']:>10.2f} {stats['std']:>10.2f} {stats['p95']:>10.2f}")
    # print(f"\n  Model VRAM (static) : {model_vram:.2f} GB")
    print(f"\n  Static VRAM (model weights) : {static_vram_gb:.2f} GB")
    print("=" * 65)

    print("\nDone.")


if __name__ == "__main__":
    main()
