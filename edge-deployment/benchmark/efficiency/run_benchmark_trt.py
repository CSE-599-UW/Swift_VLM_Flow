"""
run_benchmark_trt.py
Efficiency benchmark for TensorRT-LLM Qwen2-VL engine.

Mirrors run_benchmark.py exactly — same metrics, same JSON format,
same VQAv2 samples (via seed) — so results are directly comparable.

Metrics measured:
  - TTFT (ms)          : Time to First Token
  - Decode Latency (ms/tok)     : (Total Latency - TTFT) / output_tokens
  - Static VRAM (GB)            : GPU memory after engine load (nvidia-smi delta)
  - Dynamic VRAM (GB)           : Per-inference increment above static baseline

Usage:
    python3 run_benchmark_trt.py
    python3 run_benchmark_trt.py --num_samples 50 --warmup 3 --output_tag bf16_v1
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

# TRT-LLM runtime
from tensorrt_llm.runtime import MultimodalModelRunner

import config
import metrics
import data_loader

# ── Argument Parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TRT-LLM VLM Benchmark")
    parser.add_argument("--num_samples", type=int, default=config.NUM_SAMPLES,
                        help="Number of VQAv2 samples to evaluate")
    parser.add_argument("--warmup", type=int, default=config.NUM_WARMUP,
                        help="Number of warmup runs")
    parser.add_argument("--max_new_tokens", type=int, default=config.MAX_NEW_TOKENS,
                        help="Max tokens to generate per sample")
    parser.add_argument("--output_tag", type=str, default=None,
                        help="Optional tag appended to output filenames")
    parser.add_argument("--engine_dir", type=str,
                        default="/workspace/trt_engines/qwen2vl",
                        help="Path to TRT engine directory (must contain llm/ and vision/)")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp16", "fp8", "bf16"],
                        help="Engine precision label for reporting")
    return parser.parse_args()


# ── Minimal args object for MultimodalModelRunner ─────────────────────────────

class TRTArgs:
    """
    Minimal args namespace accepted by MultimodalModelRunner.__init__().
    Only fields actually used by the runner are set; everything else is
    left at a safe default.
    """
    def __init__(self, engine_dir, hf_model_dir, max_new_tokens):
        self.engine_dir             = engine_dir
        self.hf_model_dir           = hf_model_dir
        self.max_new_tokens         = max_new_tokens
        self.batch_size             = 1
        self.num_beams              = 1
        self.top_k                  = 1
        self.top_p                  = 0.0
        self.temperature            = 1.0
        self.repetition_penalty     = 1.0
        self.run_profiling          = False
        self.profiling_iterations   = 10
        self.check_accuracy         = False
        self.video_path             = None
        self.video_num_frames       = 8
        self.image_path             = None
        self.audio_path             = None
        self.path_sep               = ","
        self.prompt_sep             = "|"
        self.enable_context_fmha_fp32_acc = False
        self.enable_chunked_context = False
        self.mm_embedding_offloading = False
        self.session                = "cpp_llm_only"
        self.kv_cache_free_gpu_memory_fraction = 0.1
        self.cross_kv_cache_fraction = None
        self.multi_block_mode       = False
        self.lora_task_uids         = None
        self.debug_mode             = False
        self.log_level              = "error"
        self.visual_engine_name     = "model.engine"
        self.audio_engine_name      = None


# ── Model Loading ──────────────────────────────────────────────────────────────

def _get_gpu_memory_used_gb() -> float:
    """
    Query current GPU memory usage via nvidia-smi.
    More reliable than torch.cuda.max_memory_allocated() for TRT-LLM
    because TRT-LLM allocates memory outside of PyTorch's allocator.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        mib = float(result.stdout.strip().split("\n")[0])
        return round(mib / 1024, 3)
    except Exception:
        return 0.0


def load_trt_model(engine_dir: str, max_new_tokens: int):
    """
    Load TRT-LLM MultimodalModelRunner and return (model, model_vram_gb).

    VRAM is measured via nvidia-smi before and after engine load,
    because TRT-LLM allocates GPU memory outside PyTorch's allocator
    (torch.cuda.max_memory_allocated() always returns 0 for TRT engines).
    """

    vram_before = _get_gpu_memory_used_gb()

    t0 = time.perf_counter()
    args = TRTArgs(
        engine_dir=engine_dir,
        hf_model_dir=config.MODEL_PATH,
        max_new_tokens=max_new_tokens,
    )
    model = MultimodalModelRunner(args)
    load_time = time.perf_counter() - t0

    vram_after  = _get_gpu_memory_used_gb()
    static_vram_gb  = round(vram_after - vram_before, 3)

    print(f"  Engine dir    : {engine_dir}")
    print(f"  Load time     : {load_time:.2f}s")
    print(f"  VRAM before   : {vram_before:.2f} GB")
    print(f"  VRAM after    : {vram_after:.2f} GB")
    print(f"  Static VRAM    : {static_vram_gb:.2f} GB")

    return model, static_vram_gb


# ── Single-sample Inference ────────────────────────────────────────────────────

def run_single_trt(model, sample: dict, max_new_tokens: int,
                   static_vram_gb: float) -> dict:
    """
    Run TRT inference on one VQAv2 sample and return per-sample metrics.

    TTFT is approximated by running max_new_tokens=1 first, then the full
    generation — identical methodology to run_benchmark.py for fair comparison.
    """
    prompt = data_loader.format_prompt(sample["question"])
    image  = sample["image"]

    # ── TTFT: generate exactly 1 token ────────────────────────────────────────
    timer = metrics.LatencyTimer()

    timer.start()
    _, _ = model.run(
        input_text=[prompt],
        input_image=[image],
        input_audio=None,
        max_new_tokens=1,
    )
    timer.mark_first_token()

    # ── Full generation ────────────────────────────────────────────────────────
    _, output_text = model.run(
        input_text=[prompt],
        input_image=[image],
        input_audio=None,
        max_new_tokens=max_new_tokens,
    )
    timer.stop()

    # output_text is list of lists: [[token1, token2, ...]] for beam=1
    if isinstance(output_text[0], list):
        predicted_answer = output_text[0][0]
    else:
        predicted_answer = output_text[0]

    # Estimate output token count from decoded string
    output_tokens = len(model.tokenizer.encode(
        predicted_answer, add_special_tokens=False,
        allowed_special=set(),
    )) + 1

    # Dynamic VRAM: current total usage minus static baseline
    dynamic_vram_gb = round(
        max(0.0, _get_gpu_memory_used_gb() - static_vram_gb), 3)
 
    decode_latency_per_tok = metrics.compute_decode_latency_per_token(
        timer.total_latency_ms, timer.ttft_ms, output_tokens)

    return {
        "question_id":            sample["question_id"],
        "question":               sample["question"],
        "predicted_answer":       predicted_answer,
        "ground_truth_answers":   sample["answers"],
        "ttft_ms":                round(timer.ttft_ms, 3),
        "total_latency_ms":       round(timer.total_latency_ms, 3),
        "decode_latency_ms_per_tok":round(decode_latency_per_tok, 3),
        "dynamic_vram_gb":        dynamic_vram_gb,
        "output_tokens":          output_tokens,
    }


# ── Warmup ─────────────────────────────────────────────────────────────────────

def warmup(model, samples: list, num_warmup: int, max_new_tokens: int,
           static_vram_gb: float):
    print(f"\n[Step 3/5] Warming up ({num_warmup} runs)...")
    for i in range(min(num_warmup, len(samples))):
        result = run_single_trt(model, samples[i], max_new_tokens, static_vram_gb)
        print(
            f"  Warmup {i+1}/{num_warmup} | "
            f"Latency={result['total_latency_ms']:.1f}ms"
        )


# ── Main Benchmark Loop ────────────────────────────────────────────────────────

def run_benchmark(model, samples: list, max_new_tokens: int,
                  static_vram_gb: float) -> list[dict]:
    print(f"\n[Step 4/5] Benchmarking {len(samples)} samples...")
    results = []
    errors  = 0

    for i, sample in enumerate(samples):
        try:
            result = run_single_trt(model, sample, max_new_tokens, static_vram_gb)
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
        "run_id":       run_id,
        "timestamp":    datetime.now().isoformat(),
        "config": {
            "model":          config.MODEL_NAME,
            "precision":      args.precision,
            "backend":        f"tensorrt-llm-{args.precision}",
            "engine_dir":     args.engine_dir,
            "num_samples":    args.num_samples,
            "num_warmup":     args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "dataset":        config.VQAV2_DATASET,
            "split":          config.VQAV2_SPLIT,
            "seed":           config.VQAV2_SEED,
        },
        "static_vram_gb": round(static_vram_gb, 3),
        "summary":       summary,
        "per_sample":    per_sample,
    }

    tag      = f"_{args.output_tag}" if args.output_tag else ""
    filename = f"{config.RAW_DIR}/trt/{args.precision}_{run_id}{tag}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    return filename


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print(" VLM TRT Benchmark")
    print(f" Model    : {config.MODEL_NAME} ({args.precision})")
    print(f" Backend  : TensorRT-LLM")
    print(f" Engine   : {args.engine_dir}")
    print(f" Samples  : {args.num_samples} (VQAv2 {config.VQAV2_SPLIT})")
    print(f" Run ID   : {run_id}")
    print("=" * 65)

    # Step 1: Load TRT engines
    print("\n[Step 1/5] Loading TRT engines...")
    model, static_vram_gb = load_trt_model(args.engine_dir, args.max_new_tokens)

    # Step 2: Load data (same seed as baseline for fair comparison)
    print(f"\n[Step 2/5] Loading VQAv2 samples...")
    samples = data_loader.load_vqav2_samples(num_samples=args.num_samples)

    # Step 3: Warmup
    warmup(model, samples, args.warmup, args.max_new_tokens, static_vram_gb)

    # Step 4: Benchmark
    per_sample = run_benchmark(model, samples, args.max_new_tokens, static_vram_gb)

    # Step 5: Summarize + save
    print(f"\n[Step 5/5] Saving results...")
    summary  = metrics.summarize_results(per_sample)
    raw_path = save_results(per_sample, summary, static_vram_gb, args, run_id)

    # Print summary
    print("\n" + "=" * 65)
    print(" RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Mean':>10} {'Std':>10} {'p95':>10}")
    print(f"  {'-'*60}")
    for metric, stats in summary.items():
        print(f"  {metric:<30} {stats['mean']:>10.2f} "
              f"{stats['std']:>10.2f} {stats['p95']:>10.2f}")
    print(f"\n  Static VRAM (engine) : {static_vram_gb:.2f} GB")
    print("=" * 65)
    print(f"\n  Raw results → {raw_path}")


if __name__ == "__main__":
    main()