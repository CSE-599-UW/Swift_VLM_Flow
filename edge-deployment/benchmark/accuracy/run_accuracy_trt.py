"""
run_accuracy_trt.py
Accuracy benchmark for TensorRT-LLM Qwen2-VL engine.

Evaluates on VQAv2, POPE (3 splits), and MME.
Results are saved as JSON compatible with run_accuracy_baseline.py
so numbers can be compared directly.

Usage:
    # All three benchmarks
    python3 run_accuracy_trt.py

    # Specific benchmarks only
    python3 run_accuracy_trt.py --tasks vqa pope

    # Quick smoke-test
    python3 run_accuracy_trt.py --tasks vqa --vqa_samples 20

    # Custom engine path
    python3 run_accuracy_trt.py --engine_dir /workspace/trt_engines/qwen2vl_int4
"""
import math
import argparse
import json
import os
import traceback
from datetime import datetime

# TRT-LLM runtime — same as run_benchmark_trt.py
from tensorrt_llm.runtime import MultimodalModelRunner

import config_accuracy as cfg
import data_loader_acc  as dl
import accuracy_metrics as am


# ── Argument Parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="TRT-LLM VLM Accuracy Benchmark")
    parser.add_argument("--tasks", nargs="+",
                        choices=["vqa", "pope", "mme"], default=["vqa", "pope", "mme"],
                        help="Which benchmarks to run")
    parser.add_argument("--engine_dir", type=str,
                        # default="/workspace/trt_engines/qwen2vl",
                        default=cfg.ENGINE_PATH,
                        help="TRT engine directory (must contain llm/ and vision/)")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["fp16", "fp8", "bf16", "int4", "int8"],
                        help="Engine precision label (for output filename only)")
    parser.add_argument("--vqa_samples", type=int, default=cfg.VQAV2_SAMPLES)
    parser.add_argument("--pope_samples", type=int, default=cfg.POPE_SAMPLES,
                        help="Samples per POPE split (None = full split)")
    parser.add_argument("--mme_samples", type=int, default=None,
                        help="MME samples (None = full dataset)")
    parser.add_argument("--warmup", type=int, default=cfg.NUM_WARMUP)
    parser.add_argument("--output_tag", type=str, default=None)
    return parser.parse_args()


# ── TRT Args (minimal namespace for MultimodalModelRunner) ────────────────────

class TRTArgs:
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

def load_model(engine_dir: str):
    print(f"[Model] Loading TRT engine from {engine_dir} ...")
    trt_args = TRTArgs(
        engine_dir=engine_dir,
        hf_model_dir=cfg.MODEL_PATH,
        max_new_tokens=cfg.MAX_NEW_TOKENS,
    )
    model = MultimodalModelRunner(trt_args)
    print("[Model] Engine loaded.")
    return model


# ── Single Inference ───────────────────────────────────────────────────────────

def infer(model, prompt: str, image) -> str:
    """
    Run TRT inference and return the decoded answer string.
    image must be a PIL.Image (same as run_benchmark_trt.py).
    """
    image = dl.resize_img(image)
    _, output_text = model.run(
        input_text=[prompt],
        input_image=[image],
        input_audio=None,
        max_new_tokens=cfg.MAX_NEW_TOKENS,
    )
    # output_text shape: [[answer_str]] for beam=1
    if isinstance(output_text[0], list):
        return output_text[0][0]
    return output_text[0]


# ── Warmup ─────────────────────────────────────────────────────────────────────

def warmup(model, sample: dict, prompt_fn, num_warmup: int):
    print(f"[Warmup] Running {num_warmup} warmup inference(s)...")
    for i in range(num_warmup):
        prompt = prompt_fn(sample["question"])
        _ = infer(model, prompt, sample["image"])
        print(f"  warmup {i+1}/{num_warmup} done")


# ══════════════════════════════════════════════════════════════════════
# Per-task runners
# ══════════════════════════════════════════════════════════════════════

def run_vqa(model, num_samples: int) -> tuple[list[dict], dict]:
    """Run VQAv2 accuracy benchmark. Returns (per_sample, scores)."""
    print(f"\n{'='*60}")
    print(f" VQAv2  ({num_samples} samples)")
    print(f"{'='*60}")

    samples = dl.load_vqav2_samples(num_samples=num_samples)
    warmup(model, samples[0], dl.format_vqa_prompt, cfg.NUM_WARMUP)

    per_sample = []
    errors = 0
    for i, s in enumerate(samples):
        try:
            prompt = dl.format_vqa_prompt(s["question"])
            pred   = infer(model, prompt, s["image"])
            record = {
                "question_id":          s["question_id"],
                "question":             s["question"],
                "predicted_answer":     pred,
                "ground_truth_answers": s["answers"],
            }
            per_sample.append(record)
            print(f"  [{i+1:>4}/{num_samples}] pred='{pred[:40]}'"
                  f"  gt='{s['answers'][0] if s['answers'] else ''}'")
        except Exception as e:
            errors += 1
            print(f"  [{i+1:>4}/{num_samples}] ERROR: {e}")
            traceback.print_exc()

    scores = am.score_vqa(per_sample)
    am.print_vqa_summary(scores)
    print(f"  (errors: {errors})")
    return per_sample, scores


def run_pope(model, num_samples_per_split: int) -> tuple[dict, dict]:
    """Run POPE accuracy benchmark (all 3 splits). Returns (per_split_samples, scores)."""
    print(f"\n{'='*60}")
    print(f" POPE  ({num_samples_per_split} samples × 3 splits)")
    print(f"{'='*60}")

    results_by_split: dict[str, list[dict]] = {}
    all_samples_by_split: dict[str, list[dict]] = {}

    for split_name in cfg.POPE_SPLITS:
        print(f"\n  ── Split: {split_name} ──")
        samples = dl.load_pope_samples(split_name, num_samples_per_split)
        warmup(model, samples[0], dl.format_pope_prompt, cfg.NUM_WARMUP)

        per_sample = []
        errors = 0
        for i, s in enumerate(samples):
            try:
                prompt = dl.format_pope_prompt(s["question"])
                pred   = infer(model, prompt, s["image"])
                record = {
                    "question_id":      s["question_id"],
                    "question":         s["question"],
                    "predicted_answer": pred,
                    "ground_truth":     s["ground_truth"],
                    "split":            split_name,
                }
                per_sample.append(record)
                yn = am.normalize_yn(pred)
                correct = "✓" if yn == s["ground_truth"] else "✗"
                print(f"    [{i+1:>4}/{len(samples)}] {correct}"
                      f"  pred={yn:<4}  gt={s['ground_truth']}")
            except Exception as e:
                errors += 1
                print(f"    [{i+1:>4}/{len(samples)}] ERROR: {e}")
                traceback.print_exc()

        results_by_split[split_name]      = per_sample
        all_samples_by_split[split_name]  = per_sample
        print(f"  (errors: {errors})")

    scores = am.score_pope(results_by_split)
    am.print_pope_summary(scores)
    return all_samples_by_split, scores


def run_mme(model, num_samples) -> tuple[list[dict], dict]:
    """Run MME accuracy benchmark. Returns (per_sample, scores)."""
    print(f"\n{'='*60}")
    print(f" MME  ({'all' if num_samples is None else num_samples} samples)")
    print(f"{'='*60}")

    samples = dl.load_mme_samples(num_samples)
    warmup(model, samples[0], dl.format_mme_prompt, cfg.NUM_WARMUP)

    per_sample = []
    errors = 0
    # errors = []
    for i, s in enumerate(samples):
        try:
            prompt = dl.format_mme_prompt(s["question"])
            pred   = infer(model, prompt, s["image"])
            record = {
                "question_id":      s["question_id"],
                "question":         s["question"],
                "predicted_answer": pred,
                "ground_truth":     s["ground_truth"],
                "task":             s["task"],
                "image_id":         s["image_id"],
            }
            per_sample.append(record)
            yn = am.normalize_yn(pred)
            correct = "✓" if yn == s["ground_truth"] else "✗"
            print(f"  [{i+1:>4}/{len(samples)}] {correct}"
                  f"  task={s['task']:<25}  pred={yn}  gt={s['ground_truth']}")
        except Exception as e:
            errors += 1
            print(f"  [{i+1:>4}/{len(samples)}] ERROR: {e}")
            traceback.print_exc()
            # w, h = s["image"].size
            # tokens = math.ceil(h/28) * math.ceil(w/28)
            # errors.append({
            #     "idx":    i,
            #     "error":  str(e).splitlines()[0],   # 只取第一行
            #     "tokens": tokens,
            #     "size":   f"{w}x{h}",
            #     "task":   s["task"],
            # })

    scores = am.score_mme(per_sample)
    am.print_mme_summary(scores)
    print(f"  (errors: {errors})")

    # import pandas as pd
    # df = pd.DataFrame(errors)
    # print(df.groupby("error")["tokens"].describe())
    # print(df["tokens"].hist(bins=20))

    return per_sample, scores


# ── Save Results ───────────────────────────────────────────────────────────────

def save_results(payload: dict, precision: str, tag: str, run_id: str) -> str:
    out_dir = os.path.join(cfg.ACC_DIR, "trt")
    os.makedirs(out_dir, exist_ok=True)
    suffix   = f"_{tag}" if tag else ""
    filename = os.path.join(out_dir, f"{precision}_{run_id}{suffix}.json")
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return filename


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(" VLM Accuracy Benchmark  [TRT]")
    print(f" Model    : {cfg.MODEL_NAME}  ({args.precision})")
    print(f" Engine   : {args.engine_dir}")
    print(f" Tasks    : {', '.join(args.tasks)}")
    print(f" Run ID   : {run_id}")
    print("=" * 60)

    # Load TRT engine once — shared across all tasks
    model = load_model(args.engine_dir)

    output = {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(),
        "backend":   f"tensorrt-llm-{args.precision}",
        "config": {
            "model":        cfg.MODEL_NAME,
            "precision":    args.precision,
            "engine_dir":   args.engine_dir,
            "vqa_samples":  args.vqa_samples,
            "pope_samples": args.pope_samples,
            "mme_samples":  args.mme_samples,
            "max_new_tokens": cfg.MAX_NEW_TOKENS,
        },
        "results": {},
    }

    # ── VQAv2 ──────────────────────────────────────────────────────────
    if "vqa" in args.tasks:
        per_sample, scores = run_vqa(model, args.vqa_samples)
        output["results"]["vqa"] = {
            "scores":     scores,
            "per_sample": per_sample,
        }

    # ── POPE ───────────────────────────────────────────────────────────
    if "pope" in args.tasks:
        per_split_samples, scores = run_pope(model, args.pope_samples)
        output["results"]["pope"] = {
            "scores":            scores,
            "per_split_samples": per_split_samples,
        }

    # ── MME ────────────────────────────────────────────────────────────
    if "mme" in args.tasks:
        per_sample, scores = run_mme(model, args.mme_samples)
        output["results"]["mme"] = {
            "scores":     scores,
            "per_sample": per_sample,
        }

    # ── Save ────────────────────────────────────────────────────────────
    path = save_results(output, args.precision, args.output_tag, run_id)

    print(f"\n{'='*60}")
    print(" FINAL SUMMARY")
    print(f"{'='*60}")
    if "vqa" in output["results"]:
        am.print_vqa_summary(output["results"]["vqa"]["scores"])
    if "pope" in output["results"]:
        am.print_pope_summary(output["results"]["pope"]["scores"])
    if "mme" in output["results"]:
        am.print_mme_summary(output["results"]["mme"]["scores"])
    print(f"\n  Results saved → {path}")


if __name__ == "__main__":
    main()
