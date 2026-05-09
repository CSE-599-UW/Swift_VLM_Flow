"""
metrics.py
Metric computation utilities for the VLM benchmark.

Metrics computed per sample:
  - ttft_ms              : Time to First Token (milliseconds)
  - total_latency_ms     : Total generation time (milliseconds)
  - throughput_tok_per_sec: Output tokens / total generation time
  - peak_vram_gb         : Peak GPU memory allocated during inference (GB)

Aggregate statistics computed across all samples:
  - mean, std, median (p50), p95
"""

import time
import statistics
from typing import Any
import torch


# ── Per-sample Measurement ─────────────────────────────────────────────────────

class LatencyTimer:
    """
    Context manager that measures TTFT and total latency for a single
    generation call using a HuggingFace streamer-style callback.

    Usage:
        timer = LatencyTimer()
        with timer:
            output = model.generate(..., streamer=timer.streamer)
        print(timer.ttft_ms, timer.total_latency_ms)
    """

    def __init__(self):
        self._t_start: float = 0.0
        self._t_first_token: float = 0.0
        self._t_end: float = 0.0
        self._first_token_seen: bool = False

    def start(self):
        torch.cuda.synchronize()
        self._t_start = time.perf_counter()
        self._first_token_seen = False

    def mark_first_token(self):
        if not self._first_token_seen:
            torch.cuda.synchronize()
            self._t_first_token = time.perf_counter()
            self._first_token_seen = True

    def stop(self):
        torch.cuda.synchronize()
        self._t_end = time.perf_counter()

    @property
    def ttft_ms(self) -> float:
        """Time to First Token in milliseconds."""
        if self._t_first_token == 0.0:
            # Fallback: TTFT not measured (no streamer), return total latency
            return self.total_latency_ms
        return (self._t_first_token - self._t_start) * 1000

    @property
    def total_latency_ms(self) -> float:
        """Total generation latency in milliseconds."""
        return (self._t_end - self._t_start) * 1000


def measure_vram_gb() -> float:
    """Return peak GPU memory allocated since last reset, in GB."""
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_vram_stats():
    """Reset the peak memory tracking counter."""
    torch.cuda.reset_peak_memory_stats()


def compute_throughput(num_output_tokens: int, total_latency_ms: float) -> float:
    """
    Compute throughput in tokens per second.

    Args:
        num_output_tokens : Number of newly generated tokens.
        total_latency_ms  : Total generation time in milliseconds.

    Returns:
        Throughput in tokens/sec. Returns 0.0 if latency is zero.
    """
    if total_latency_ms <= 0:
        return 0.0
    return num_output_tokens / (total_latency_ms / 1000)


# ── Aggregate Statistics ───────────────────────────────────────────────────────

def aggregate(values: list[float]) -> dict[str, float]:
    """
    Compute summary statistics for a list of per-sample metric values.

    Returns:
        Dict with keys: mean, std, median, p95, min, max.
    """
    if not values:
        return {"mean": 0, "std": 0, "median": 0, "p95": 0, "min": 0, "max": 0}

    sorted_vals = sorted(values)
    p95_idx = int(len(sorted_vals) * 0.95)

    return {
        "mean":   round(statistics.mean(values), 3),
        "std":    round(statistics.stdev(values) if len(values) > 1 else 0.0, 3),
        "median": round(statistics.median(values), 3),
        "p95":    round(sorted_vals[min(p95_idx, len(sorted_vals) - 1)], 3),
        "min":    round(min(values), 3),
        "max":    round(max(values), 3),
    }


def summarize_results(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Given a list of per-sample result dicts, compute aggregate statistics
    for all numeric metrics.

    Args:
        per_sample: List of dicts with keys matching METRICS in config.py.

    Returns:
        Dict mapping each metric name to its aggregate statistics dict.
    """
    metric_keys = [
        "ttft_ms",
        "total_latency_ms",
        "throughput_tok_per_sec",
        "peak_vram_gb",
        "output_tokens",
    ]

    summary = {}
    for key in metric_keys:
        values = [s[key] for s in per_sample if key in s and s[key] is not None]
        summary[key] = aggregate(values)

    return summary
