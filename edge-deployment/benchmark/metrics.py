"""
metrics.py
Metric computation utilities for the VLM benchmark.

Metrics computed per sample:
  - ttft_ms              : Time to First Token (milliseconds)
  - total_latency_ms     : Total generation time (milliseconds)
  - decode_latency_ms_per_tok: Decode-phase time per output token (ms/tok)
                           = (total_latency_ms - ttft_ms) / output_tokens
  - dynamic_vram_gb      : Per-inference VRAM increment (GB)
                           = peak during inference − static baseline
                           Measured by resetting peak stats just before each
                           generate() call and reading max_memory_allocated()
                           after. Reflects KV-cache + activation overhead only.
  - static_vram_gb       : VRAM occupied after model weights are loaded,
                           before any inference begins. Determines whether the
                           model fits on the target device.

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


# =================================================================================
# Measure VRAM
# =================================================================================
def _bytes_to_gb(n: int) -> float:
    return n / (1024 ** 3)

def reset_vram_stats():
    """Reset the peak memory tracking counter."""
    torch.cuda.reset_peak_memory_stats()

# Static VRAM
def measure_static_vram_gb() -> float:
    """
    Capture static VRAM: the memory currently allocated by PyTorch (i.e. model
    weights sitting in VRAM).  Call this once after model.eval() and before any
    inference begins.
 
    Uses memory_allocated() rather than max_memory_allocated() so the reading
    is not inflated by any earlier peak (e.g. the transient spike during weight
    loading).
    """
    torch.cuda.synchronize()
    return _bytes_to_gb(torch.cuda.memory_allocated())

# Dynamic VRAM
def measure_dynamic_vram_gb(static_vram_gb: float) -> float:
    """
    Capture the per-inference VRAM increment: how much extra memory was
    allocated on top of the static baseline during a single generate() call.
    This reflects KV-cache growth plus intermediate activation buffers.
 
    Args:
        static_vram_gb: The static baseline measured once after model load
                        (from measure_static_vram_gb()).
 
    Returns:
        Dynamic VRAM increment in GB (≥ 0).  Clamped to 0 in the unlikely
        event that the peak reading falls below the static baseline due to
        rounding.
    """
    torch.cuda.synchronize()
    peak_gb = _bytes_to_gb(torch.cuda.max_memory_allocated())
    return max(0.0, peak_gb - static_vram_gb)



# =================================================================================

def compute_decode_latency_per_token(
    total_latency_ms: float,
    ttft_ms: float,
    num_output_tokens: int,
) -> float:
    """
    Compute per-token decode latency in milliseconds.
 
    Decode time is isolated by subtracting TTFT (prefill) from total latency,
    then divided by the number of output tokens:
 
        decode_latency_ms_per_tok = (total_latency_ms - ttft_ms) / output_tokens
 
    TTFT comes from the first generate() call (max_new_tokens=1, pure prefill).
    Total latency comes from the second generate() call (full decode).
    Because both calls use identical inputs, their prefill times are equivalent,
    so the subtraction correctly isolates the decode phase of the full run.
 
    Args:
        total_latency_ms  : Total generation time of the full generate() call (ms).
        ttft_ms           : Time-to-first-token from the single-token generate() call (ms).
        num_output_tokens : Number of newly generated tokens in the full run.
 
    Returns:
        Decode latency per token in ms/tok. Returns 0.0 if output_tokens is 0.
    """
    if num_output_tokens <= 0:
        return 0.0
    decode_ms = max(0.0, total_latency_ms - ttft_ms)
    
    return decode_ms / num_output_tokens


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
        "decode_latency_ms_per_tok",
        "dynamic_vram_gb",
        "output_tokens",
    ]

    summary = {}
    for key in metric_keys:
        values = [s[key] for s in per_sample if key in s and s[key] is not None]
        summary[key] = aggregate(values)

    return summary
