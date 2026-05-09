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
from typing import Optional
from datasets import load_dataset
from PIL import Image

import config


def load_vqav2_samples(
    num_samples: int = config.NUM_SAMPLES,
    seed: int = config.VQAV2_SEED,
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