"""
accuracy_metrics.py
Scoring logic for VQAv2, POPE, and MME.

Each task has its own canonical scoring function that matches what
lmms-eval / the original papers report, so your numbers are directly
comparable to the baseline lmms-eval run.
"""

import re
import statistics
from collections import defaultdict
from typing import Any


# ══════════════════════════════════════════════════════════════════════
# Shared utility
# ══════════════════════════════════════════════════════════════════════

def normalize_answer(text: str) -> str:
    """
    VQAv2 official answer normalization.
    Mirrors vqaEval.py from GT-Vision-Lab/VQA.
    """
    text = text.lower().strip()
    # punctuation
    text = re.sub(r"([.,!?\"';:)(])", " ", text)
    # articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am",
    }
    for c, expansion in contractions.items():
        text = text.replace(c, expansion)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_yn(text: str) -> str:
    """Extract yes/no from model output (handles 'Yes.', 'No,', 'yes' etc.)."""
    text = text.lower().strip().rstrip(".,!?")
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    # Fallback: scan for the word
    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"
    return text   # keep raw if neither found


# ══════════════════════════════════════════════════════════════════════
# VQAv2
# ══════════════════════════════════════════════════════════════════════

def vqa_soft_score(predicted: str, ground_truths: list[str]) -> float:
    """
    VQAv2 soft accuracy per sample.
    Score = min(# annotators who gave this answer / 3, 1.0)
    Annotators: typically 10 per question (ground_truths list has 10 items).
    """
    pred_norm = normalize_answer(predicted)
    matches = sum(normalize_answer(gt) == pred_norm for gt in ground_truths)
    return min(matches / 3.0, 1.0)


def score_vqa(per_sample: list[dict]) -> dict:
    """
    Compute VQAv2 accuracy over all samples.

    Args:
        per_sample: list of dicts with keys:
            predicted_answer (str), ground_truth_answers (list[str])

    Returns:
        {
          "accuracy":   float   # 0–100, main reported number
          "num_samples": int
          "per_sample_scores": list[float]   # 0.0–1.0 per question
        }
    """
    scores = []
    for s in per_sample:
        score = vqa_soft_score(s["predicted_answer"], s["ground_truth_answers"])
        s["accuracy_score"] = round(score, 4)
        scores.append(score)

    mean_acc = statistics.mean(scores) * 100 if scores else 0.0
    return {
        "accuracy":          round(mean_acc, 2),
        "num_samples":       len(scores),
        "per_sample_scores": scores,
    }


# ══════════════════════════════════════════════════════════════════════
# POPE
# ══════════════════════════════════════════════════════════════════════

def score_pope_split(per_sample: list[dict]) -> dict:
    """
    POPE metrics for one split (random / popular / adversarial).

    Ground truth is 'yes' or 'no' (lowercase).
    Model output is normalized to 'yes'/'no' via normalize_yn().

    Reported metrics (matching lmms-eval):
        accuracy  : (TP + TN) / total
        precision : TP / (TP + FP)
        recall    : TP / (TP + FN)    [= sensitivity]
        f1        : harmonic mean of precision and recall
        yes_ratio : fraction of predictions that are 'yes'
    """
    tp = tn = fp = fn = 0
    yes_preds = 0

    for s in per_sample:
        pred = normalize_yn(s["predicted_answer"])
        gt   = s["ground_truth"].lower().strip()

        s["pred_normalized"] = pred

        if pred == "yes":
            yes_preds += 1

        if gt == "yes":
            if pred == "yes":
                tp += 1
            else:
                fn += 1
        else:  # gt == "no"
            if pred == "no":
                tn += 1
            else:
                fp += 1

    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total * 100 if total else 0.0
    precision = tp / (tp + fp) * 100    if (tp + fp) else 0.0
    recall    = tp / (tp + fn) * 100    if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    yes_ratio = yes_preds / total       if total else 0.0

    return {
        "accuracy":   round(accuracy,  2),
        "precision":  round(precision, 2),
        "recall":     round(recall,    2),
        "f1":         round(f1,        2),
        "yes_ratio":  round(yes_ratio, 4),
        "num_samples": total,
    }


def score_pope(results_by_split: dict[str, list[dict]]) -> dict:
    """
    Score all POPE splits and compute the macro-average F1.

    Args:
        results_by_split: {"random": [...], "popular": [...], "adversarial": [...]}

    Returns:
        {
          "per_split":  {"random": {...}, "popular": {...}, "adversarial": {...}},
          "avg_accuracy": float,
          "avg_f1":       float,
        }
    """
    per_split = {}
    for split_name, samples in results_by_split.items():
        per_split[split_name] = score_pope_split(samples)

    f1s  = [v["f1"]       for v in per_split.values()]
    accs = [v["accuracy"] for v in per_split.values()]

    return {
        "per_split":    per_split,
        "avg_accuracy": round(statistics.mean(accs), 2) if accs else 0.0,
        "avg_f1":       round(statistics.mean(f1s),  2) if f1s  else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════
# MME
# ══════════════════════════════════════════════════════════════════════

def score_mme(per_sample: list[dict]) -> dict:
    """
    MME scoring — matches the official MME evaluation script.

    MME groups questions into tasks (e.g. 'color', 'count', 'commonsense_reasoning').
    Each image has exactly 2 questions (a positive and a negative).
    Per-task score = sum over image pairs of:
        2  if both answers correct
        1  if exactly one correct
        0  if neither correct
    Max score per task = 2 × number of images in that task.

    Perception tasks  (14): contribute to perception_score
    Cognition tasks   (4):  contribute to cognition_score
    Total score = perception_score + cognition_score

    Args:
        per_sample: list of dicts with keys:
            task (str), image_id (str), predicted_answer (str), ground_truth (str)

    Returns:
        {
          "total_score":       float
          "perception_score":  float
          "cognition_score":   float
          "per_task":          {task_name: {"score": float, "max_score": float, "acc": float}}
        }
    """
    PERCEPTION_TASKS = {
        "existence", "count", "position", "color", "posters", "celebrity",
        "scene", "landmark", "artwork", "OCR",
        "commonsense_reasoning", "numerical_calculation",
        "text_translation", "code_reasoning",
    }
    # Everything not in PERCEPTION_TASKS is treated as cognition
    # (in practice: "commonsense_reasoning", "numerical_calculation",
    #               "text_translation", "code_reasoning" are cognition)
    COGNITION_TASKS = {
        "commonsense_reasoning", "numerical_calculation",
        "text_translation", "code_reasoning",
    }
    # Re-define perception as all tasks minus cognition
    # (keeps forward-compatibility if new tasks are added)

    # Group by task → image_id → list of samples
    task_image: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for s in per_sample:
        task_image[s["task"]][s["image_id"]].append(s)

    per_task: dict[str, dict] = {}
    perception_score = 0.0
    cognition_score  = 0.0

    for task, images in task_image.items():
        task_score = 0
        task_max   = 0
        correct_q  = 0
        total_q    = 0

        for image_id, qs in images.items():
            task_max += 2
            pair_correct = 0
            for q in qs:
                pred = normalize_yn(q["predicted_answer"])
                gt   = q["ground_truth"].lower().strip()
                q["pred_normalized"] = pred
                if pred == gt:
                    pair_correct += 1
                    correct_q    += 1
                total_q += 1
            task_score += pair_correct   # 0, 1, or 2

        acc = correct_q / total_q * 100 if total_q else 0.0
        per_task[task] = {
            "score":     task_score,
            "max_score": task_max,
            "accuracy":  round(acc, 2),
        }

        if task in COGNITION_TASKS:
            cognition_score  += task_score
        else:
            perception_score += task_score

    total_score = perception_score + cognition_score

    return {
        "total_score":      round(total_score,      2),
        "perception_score": round(perception_score, 2),
        "cognition_score":  round(cognition_score,  2),
        "per_task":         per_task,
    }


# ══════════════════════════════════════════════════════════════════════
# Pretty-print helpers
# ══════════════════════════════════════════════════════════════════════

def print_vqa_summary(result: dict):
    print("\n── VQAv2 ─────────────────────────────────────────")
    print(f"  Accuracy  : {result['accuracy']:.2f}%")
    print(f"  Samples   : {result['num_samples']}")


def print_pope_summary(result: dict):
    print("\n── POPE ──────────────────────────────────────────")
    for split, metrics in result["per_split"].items():
        print(f"  [{split:<12}]  Acc={metrics['accuracy']:.2f}%  "
              f"F1={metrics['f1']:.2f}%  "
              f"Precision={metrics['precision']:.2f}%  "
              f"Recall={metrics['recall']:.2f}%")
    print(f"  Avg Accuracy : {result['avg_accuracy']:.2f}%")
    print(f"  Avg F1       : {result['avg_f1']:.2f}%")


def print_mme_summary(result: dict):
    print("\n── MME ───────────────────────────────────────────")
    print(f"  Total Score      : {result['total_score']:.1f}")
    print(f"  Perception Score : {result['perception_score']:.1f}")
    print(f"  Cognition Score  : {result['cognition_score']:.1f}")
    print(f"  {'Task':<30} {'Score':>8}  {'Acc':>7}")
    print(f"  {'-'*48}")
    for task, stats in sorted(result["per_task"].items()):
        print(f"  {task:<30} {stats['score']:>4}/{stats['max_score']:<4}"
              f"  {stats['accuracy']:>6.2f}%")
