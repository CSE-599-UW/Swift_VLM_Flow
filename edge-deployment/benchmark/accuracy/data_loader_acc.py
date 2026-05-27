"""
data_loader_acc.py
Unified data loader for all accuracy benchmarks: VQAv2, POPE, MME.

All three loaders depend only on config_accuracy.py — no dependency on
the efficiency benchmark's config.py or data_loader.py.

Return schema per benchmark:
  VQAv2 : {image, question, answers (list[str]), question_id}
  POPE  : {image, question, ground_truth ("yes"/"no"), question_id, split}
  MME   : {image, question, ground_truth ("yes"/"no"), task, image_id, question_id}
"""

import itertools
import math
from typing import Optional
from datasets import load_dataset
from PIL import Image

import config_accuracy as cfg


# ══════════════════════════════════════════════════════════════════════
# Shared helper
# ══════════════════════════════════════════════════════════════════════

def _to_pil(image) -> Image.Image:
    """Normalise whatever HuggingFace gives us to a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")

def resize_img(
    image: Image.Image,
    max_tokens: int = cfg.TRT_MAX_VISUAL_TOKENS,
    min_tokens: int = 128,
    patch: int = 14,
    merge: int = 2,
) -> Image.Image:
    """
    Resize a PIL image so its Qwen2-VL visual token count stays within the
    TRT engine's max_multimodal_len budget before inference.
 
    Qwen2-VL token count = ceil(H / (patch*merge)) * ceil(W / (patch*merge))
                         = ceil(H/28) * ceil(W/28)
 
    The engine was built with --max_multimodal_len=4096 and
    --max_hw_dims "2128 2128" (~5776 tokens).  We cap at max_tokens=4096
    to stay within the LLM-side budget (max_multimodal_len), which is the
    tighter of the two constraints.
 
    Aspect ratio is preserved. Dimensions are rounded down to the nearest
    multiple of (patch*merge)=28 so the grid is always exact.
 
    Only called from run_accuracy_trt.py — the HF baseline never needs this.
    """
    grid = patch * merge  # 28
    w, h = image.size
    current_tokens = math.ceil(h / grid) * math.ceil(w / grid)

    # 上限：縮小
    if current_tokens > max_tokens:
        max_area = max_tokens * grid * grid
        scale    = math.sqrt(max_area / (h * w))
        new_h    = max(int(h * scale) // grid * grid, grid)
        new_w    = max(int(w * scale) // grid * grid, grid)
        image    = image.resize((new_w, new_h), Image.LANCZOS)
        new_tokens = math.ceil(new_h / grid) * math.ceil(new_w / grid)
        print(f"    [resize_for_trt] shrink {w}x{h} ({current_tokens}tok) "
              f"→ {new_w}x{new_h} ({new_tokens}tok)")

    # 下限：pad to min grid size
    elif current_tokens < min_tokens:
        min_side = math.ceil(math.sqrt(min_tokens)) * grid  # 最小邊長
        new_w    = max(w, min_side)
        new_h    = max(h, min_side)
        padded   = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        padded.paste(image, (0, 0))
        new_tokens = math.ceil(new_h / grid) * math.ceil(new_w / grid)
        print(f"    [resize_for_trt] pad    {w}x{h} ({current_tokens}tok) "
              f"→ {new_w}x{new_h} ({new_tokens}tok)")
        image = padded

    return image
 
    # if current_tokens <= max_tokens:
    #     return image  # already within budget, no-op
 
    # # Scale both sides uniformly so token count ≤ max_tokens.
    # # tokens ≈ (H/grid) * (W/grid) = (H*W) / grid²
    # # → max pixel area = max_tokens * grid²
    # max_area = max_tokens * grid * grid
    # scale    = math.sqrt(max_area / (h * w))
    # new_h    = int(h * scale) // grid * grid  # round down to grid multiple
    # new_w    = int(w * scale) // grid * grid
 
    # # Ensure at least one grid cell on each side
    # new_h = max(new_h, grid)
    # new_w = max(new_w, grid)
 
    # resized = image.resize((new_w, new_h), Image.LANCZOS)
    # new_tokens = math.ceil(new_h / grid) * math.ceil(new_w / grid)
    # print(f"    [resize_for_trt] {w}x{h} ({current_tokens}tok) "
    #       f"→ {new_w}x{new_h} ({new_tokens}tok)")
    # return resized

# ══════════════════════════════════════════════════════════════════════
# VQAv2
# ══════════════════════════════════════════════════════════════════════

def load_vqav2_samples(
    num_samples: int = cfg.VQAV2_SAMPLES,
    seed: int = cfg.VQAV2_SEED,
    split: str = cfg.VQAV2_SPLIT,
) -> list[dict]:
    """
    Load a reproducible subset of VQAv2 validation samples via streaming.

    Uses the same seed as the efficiency benchmark (cfg.VQAV2_SEED = 42)
    so the 500 samples are identical across both benchmark suites.
    """
    print(f"[DataLoader-VQA] Streaming {cfg.VQAV2_DATASET} ({split}) ...")

    dataset = load_dataset(
        cfg.VQAV2_DATASET,
        split=split,
        trust_remote_code=True,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=1000)
    subset  = list(itertools.islice(dataset, num_samples))

    samples = []
    for item in subset:
        raw = item.get("answers", [])
        if raw and isinstance(raw[0], dict):
            answers = [a["answer"] for a in raw]
        else:
            answers = list(raw)

        samples.append({
            "image":       _to_pil(item["image"]),
            "question":    item["question"],
            "answers":     answers,
            "question_id": item.get("question_id", -1),
        })

    print(f"[DataLoader-VQA] Loaded {len(samples)} samples (seed={seed})")
    return samples


# ══════════════════════════════════════════════════════════════════════
# POPE
# ══════════════════════════════════════════════════════════════════════

def load_pope_samples(
    split_name: str,
    num_samples: Optional[int] = cfg.POPE_SAMPLES,
) -> list[dict]:
    """
    Load POPE samples for one adversarial split: "random" | "popular" | "adversarial".

    lmms-lab/POPE has a 'category' column we filter on.
    Ground truth is normalised to lowercase "yes" / "no".
    """
    print(f"[DataLoader-POPE] Loading split='{split_name}' ...")

    dataset = load_dataset(
        cfg.POPE_DATASET,
        split="test",           # POPE only has a test split on HF
        trust_remote_code=True,
        streaming=True,
    )
    dataset = dataset.filter(lambda x: x.get("category", "") == split_name)

    subset = list(itertools.islice(dataset, num_samples)) if num_samples else list(dataset)

    samples = []
    for i, item in enumerate(subset):
        gt = item.get("label", item.get("answer", "")).lower().strip()
        samples.append({
            "image":        _to_pil(item["image"]),
            "question":     item["question"],
            "ground_truth": gt,
            "question_id":  item.get("question_id", i),
            "split":        split_name,
        })

    print(f"[DataLoader-POPE] Loaded {len(samples)} samples (split={split_name})")
    return samples


def load_all_pope_splits(
    splits: list[str] = cfg.POPE_SPLITS,
    num_samples_per_split: Optional[int] = cfg.POPE_SAMPLES,
) -> dict[str, list[dict]]:
    """Load all three POPE splits. Returns {split_name: [samples]}."""
    return {s: load_pope_samples(s, num_samples_per_split) for s in splits}


# ══════════════════════════════════════════════════════════════════════
# MME
# ══════════════════════════════════════════════════════════════════════

def load_mme_samples(
    num_samples: Optional[int] = cfg.MME_SAMPLES,
) -> list[dict]:
    """
    Load MME samples (perception + cognition tasks).

    image_id is used by the MME scorer to pair the two questions per image.
    When the dataset doesn't expose an explicit id we fall back to hashing
    the raw pixel bytes — stable within a single run.
    """
    print(f"[DataLoader-MME] Loading {cfg.MME_DATASET} ({cfg.MME_SPLIT}) ...")

    dataset = load_dataset(
        cfg.MME_DATASET,
        split=cfg.MME_SPLIT,
        trust_remote_code=True,
        streaming=True,
    )

    subset = list(itertools.islice(dataset, num_samples)) if num_samples else list(dataset)

    samples = []
    for i, item in enumerate(subset):
        image = _to_pil(item["image"])
        image_id = (
            item.get("image_id")
            or item.get("filename")
            or item.get("file_name")
            or str(hash(image.tobytes()))
        )
        gt   = item.get("answer", item.get("label", "")).lower().strip()
        task = item.get("category", item.get("task", "unknown"))

        samples.append({
            "image":        image,
            "question":     item["question"],
            "ground_truth": gt,
            "task":         task,
            "image_id":     str(image_id),
            "question_id":  item.get("question_id", i),
        })

    print(f"[DataLoader-MME] Loaded {len(samples)} samples")
    return samples


# ══════════════════════════════════════════════════════════════════════
# Prompt formatting
# ══════════════════════════════════════════════════════════════════════

def format_vqa_prompt(question: str) -> str:
    return cfg.VQAV2_PROMPT.format(question=question)

def format_pope_prompt(question: str) -> str:
    return cfg.POPE_PROMPT.format(question=question)

def format_mme_prompt(question: str) -> str:
    return cfg.MME_PROMPT.format(question=question)
