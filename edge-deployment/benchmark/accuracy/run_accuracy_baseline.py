"""
run_accuracy_baseline.py
Accuracy benchmark for the original HuggingFace Qwen2-VL model.

Produces results in the exact same JSON format as run_accuracy_trt.py
so you can diff the two files directly to measure accuracy degradation
from TRT compression.

Usage:
    python3 run_accuracy_baseline.py
    python3 run_accuracy_baseline.py --tasks vqa --vqa_samples 20
"""

import argparse
import json
import os
import traceback
from datetime import datetime

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import config_accuracy as cfg
import data_loader_acc  as dl
import accuracy_metrics as am


# ── Argument Parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="HF VLM Accuracy Benchmark (baseline)")
    parser.add_argument("--tasks", nargs="+",
                        choices=["vqa", "pope", "mme"], default=["vqa", "pope", "mme"])
    parser.add_argument("--model_path", type=str, default=cfg.MODEL_PATH)
    parser.add_argument("--vqa_samples",  type=int, default=cfg.VQAV2_SAMPLES)
    parser.add_argument("--pope_samples", type=int, default=cfg.POPE_SAMPLES)
    parser.add_argument("--mme_samples",  type=int, default=None)
    parser.add_argument("--warmup",       type=int, default=cfg.NUM_WARMUP)
    parser.add_argument("--output_tag",   type=str, default=None)
    return parser.parse_args()


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model(model_path: str):
    print(f"[Model] Loading HF model from {model_path} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    print("[Model] HF model loaded.")
    return model, processor


# ── Single Inference ───────────────────────────────────────────────────────────

def infer(model, processor, prompt: str, image) -> str:
    """
    Run HF inference on one (prompt, PIL image) pair and return decoded text.
    Uses the same chat-template approach as run_benchmark.py for consistency.
    """
    image = dl.resize_img(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.MAX_NEW_TOKENS,
        )

    # Strip input tokens from output
    generated_ids_trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


# ── Warmup ─────────────────────────────────────────────────────────────────────

def warmup(model, processor, sample: dict, prompt_fn, num_warmup: int):
    print(f"[Warmup] Running {num_warmup} warmup inference(s)...")
    for i in range(num_warmup):
        prompt = prompt_fn(sample["question"])
        _ = infer(model, processor, prompt, sample["image"])
        print(f"  warmup {i+1}/{num_warmup} done")


# ══════════════════════════════════════════════════════════════════════
# Per-task runners (same structure as run_accuracy_trt.py)
# ══════════════════════════════════════════════════════════════════════

def run_vqa(model, processor, num_samples: int):
    print(f"\n{'='*60}\n VQAv2  ({num_samples} samples)\n{'='*60}")
    samples = dl.load_vqav2_samples(num_samples=num_samples)
    warmup(model, processor, samples[0], dl.format_vqa_prompt, cfg.NUM_WARMUP)

    per_sample = []
    for i, s in enumerate(samples):
        try:
            prompt = dl.format_vqa_prompt(s["question"])
            pred   = infer(model, processor, prompt, s["image"])
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
            print(f"  [{i+1:>4}/{num_samples}] ERROR: {e}")
            traceback.print_exc()

    scores = am.score_vqa(per_sample)
    am.print_vqa_summary(scores)
    return per_sample, scores


def run_pope(model, processor, num_samples_per_split: int):
    print(f"\n{'='*60}\n POPE  ({num_samples_per_split} × 3 splits)\n{'='*60}")
    results_by_split = {}
    all_samples      = {}

    for split_name in cfg.POPE_SPLITS:
        print(f"\n  ── Split: {split_name} ──")
        samples = dl.load_pope_samples(split_name, num_samples_per_split)
        warmup(model, processor, samples[0], dl.format_pope_prompt, cfg.NUM_WARMUP)

        per_sample = []
        for i, s in enumerate(samples):
            try:
                prompt = dl.format_pope_prompt(s["question"])
                pred   = infer(model, processor, prompt, s["image"])
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
                print(f"    [{i+1:>4}/{len(samples)}] ERROR: {e}")
                traceback.print_exc()

        results_by_split[split_name] = per_sample
        all_samples[split_name]      = per_sample

    scores = am.score_pope(results_by_split)
    am.print_pope_summary(scores)
    return all_samples, scores


def run_mme(model, processor, num_samples):
    print(f"\n{'='*60}\n MME  ({'all' if num_samples is None else num_samples})\n{'='*60}")
    samples = dl.load_mme_samples(num_samples)
    warmup(model, processor, samples[0], dl.format_mme_prompt, cfg.NUM_WARMUP)

    per_sample = []
    for i, s in enumerate(samples):
        try:
            prompt = dl.format_mme_prompt(s["question"])
            pred   = infer(model, processor, prompt, s["image"])
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
            print(f"  [{i+1:>4}/{len(samples)}] ERROR: {e}")
            traceback.print_exc()

    scores = am.score_mme(per_sample)
    am.print_mme_summary(scores)
    return per_sample, scores


# ── Save ───────────────────────────────────────────────────────────────────────

def save_results(payload: dict, tag: str, run_id: str) -> str:
    out_dir = os.path.join(cfg.ACC_DIR, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    suffix   = f"_{tag}" if tag else ""
    filename = os.path.join(out_dir, f"bf16_{run_id}{suffix}.json")
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return filename


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(" VLM Accuracy Benchmark  [HF Baseline]")
    print(f" Model  : {cfg.MODEL_NAME}  (bf16 / HuggingFace)")
    print(f" Tasks  : {', '.join(args.tasks)}")
    print(f" Run ID : {run_id}")
    print("=" * 60)

    model, processor = load_model(args.model_path)

    output = {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(),
        "backend":   "huggingface-bf16",
        "config": {
            "model":          cfg.MODEL_NAME,
            "model_path":     args.model_path,
            "vqa_samples":    args.vqa_samples,
            "pope_samples":   args.pope_samples,
            "mme_samples":    args.mme_samples,
            "max_new_tokens": cfg.MAX_NEW_TOKENS,
        },
        "results": {},
    }

    if "vqa" in args.tasks:
        per_sample, scores = run_vqa(model, processor, args.vqa_samples)
        output["results"]["vqa"] = {"scores": scores, "per_sample": per_sample}

    if "pope" in args.tasks:
        per_split, scores = run_pope(model, processor, args.pope_samples)
        output["results"]["pope"] = {"scores": scores, "per_split_samples": per_split}

    if "mme" in args.tasks:
        per_sample, scores = run_mme(model, processor, args.mme_samples)
        output["results"]["mme"] = {"scores": scores, "per_sample": per_sample}

    path = save_results(output, args.output_tag, run_id)

    print(f"\n{'='*60}\n FINAL SUMMARY\n{'='*60}")
    if "vqa" in output["results"]:
        am.print_vqa_summary(output["results"]["vqa"]["scores"])
    if "pope" in output["results"]:
        am.print_pope_summary(output["results"]["pope"]["scores"])
    if "mme" in output["results"]:
        am.print_mme_summary(output["results"]["mme"]["scores"])
    print(f"\n  Results saved → {path}")


if __name__ == "__main__":
    main()
