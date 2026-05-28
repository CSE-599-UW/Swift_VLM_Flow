"""
data_loader.py
Loads and prepares a reproducible subset of VQAv2 samples for benchmarking.
Uses streaming mode to avoid downloading the full dataset (~100GB) to disk.

Each sample contains:
  - image      : PIL.Image
  - question   : str
  - answers    : list[str]  (ground-truth answers from VQAv2)
  - question_id: int
"""

import itertools
# import random
from typing import Optional
from datasets import load_dataset
from PIL import Image

import config

# Qwen2-VL dynamic resolution: cap longer side to limit visual token count.
#   448px  -> ~256 visual tokens  (safe, lower quality)
#   672px  -> ~576 visual tokens  (good balance for 16GB GPU)  <-- default
#   1008px -> ~1296 visual tokens (may OOM on 16GB)
MAX_IMAGE_SIZE = 672
 
VALID_CATEGORIES = {"conv", "detail", "complex", "all"}
 
 
def _resize_image(img: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    """Downscale so the longer side <= max_size. Keeps aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
 
 
def load_llava_bench_samples(
    num_samples: int = config.NUM_SAMPLES,
    seed: int = config.SAMPLE_SEED,
    category: str = "all",  # 'complex' | 'detail' | 'conv' | 'all'
    dataset_name: str = "lmms-lab/llava-bench-in-the-wild",
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """
    Load a reproducible subset of LLaVA-Bench (In-the-Wild) samples.
 
    Dataset: lmms-lab/llava-bench-in-the-wild (Parquet, 60 samples total)
    Split  : train (only split available)
 
    Category options:
      'complex'  ~20 samples, long outputs (~150-200 tokens) -- best for decode latency
      'detail'   ~20 samples, medium outputs (~100-150 tokens)
      'conv'     ~20 samples, shorter outputs (~50-100 tokens)
      'all'      all 60 samples, no filter
 
    Images are resized to MAX_IMAGE_SIZE on the longer side to prevent OOM
    on Qwen2-VL's dynamic resolution tokeniser.
    """
    assert category in VALID_CATEGORIES, \
        f"category must be one of {VALID_CATEGORIES}, got '{category}'"
 
    print(f"[DataLoader] Loading {dataset_name} (split=train, category={category})...")
 
    dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
 
    if category != "all":
        dataset = dataset.filter(lambda x: x.get("category", "") == category)
 
    all_items = list(dataset)
    # rng = random.Random(seed)
    # rng.shuffle(all_items)
    subset = all_items[:num_samples]
 
    samples = []
    for idx, item in enumerate(subset):
        gpt_answer = item.get("gpt_answer", "")
        answers = [gpt_answer] if gpt_answer else []
 
        samples.append({
            "image":       _resize_image(item["image"].convert("RGB")),
            "question":    item["question"],
            "answers":     answers,
            "question_id": item.get("question_id", idx),
            "category":    item.get("category", category),
        })
 
    print(f"[DataLoader] Loaded {len(samples)} samples "
          f"(category={category}, seed={seed}, max_image_size={MAX_IMAGE_SIZE})")
    return samples
 
#  ===================================================================================

def load_vqav2_samples(
    num_samples: int = config.NUM_SAMPLES,
    seed: int = config.SAMPLE_SEED,
    split: str = config.VQAV2_SPLIT,
    dataset_name: str = config.VQAV2_DATASET,
    cache_dir: Optional[str] = None,
) -> list[dict]:
    """
    Load a reproducible subset of VQAv2 samples using streaming mode.
    Streaming avoids downloading the full dataset to disk.

    Args:
        num_samples : Number of samples to return.
        seed        : Random seed for reproducibility.
        split       : Dataset split ('validation' recommended).
        dataset_name: HuggingFace dataset identifier.
        cache_dir   : Optional local cache directory.

    Returns:
        List of dicts with keys: image, question, answers, question_id.
    """
    print(f"[DataLoader] Streaming {dataset_name} ({split} split)...")

    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
        streaming=True,
    )

    # Shuffle with fixed seed for reproducibility, then take first N samples
    dataset = dataset.shuffle(seed=seed, buffer_size=1000)
    subset = list(itertools.islice(dataset, num_samples))

    samples = []
    for item in subset:
        raw_answers = item.get("answers", [])
        if isinstance(raw_answers, list) and len(raw_answers) > 0:
            if isinstance(raw_answers[0], dict):
                answers = [a["answer"] for a in raw_answers]
            else:
                answers = raw_answers
        else:
            answers = []

        samples.append({
            "image":       item["image"].convert("RGB"),
            "question":    item["question"],
            "answers":     answers,
            "question_id": item.get("question_id", -1),
        })

    print(f"[DataLoader] Loaded {len(samples)} samples (seed={seed}, streaming=True)")
    return samples


def format_prompt(question: str) -> str:
    """Apply the prompt template from config to a raw VQAv2 question."""
    return config.USER_PROMPT_TEMPLATE.format(question=question)