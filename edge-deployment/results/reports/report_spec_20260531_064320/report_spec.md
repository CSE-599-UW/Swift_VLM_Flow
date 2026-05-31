# Qwen2.5-VL-7B EAGLE3 Speculative Decoding — TensorRT-Edge-LLM (GB10, bf16)

Decode speedup from EAGLE3 speculative decoding vs the same base model decoding
autoregressively. LLaVA-Bench-in-the-Wild, 60 samples, greedy.

| Metric | Base (SD-off) | EAGLE3 (SD-on) | Gain |
|---|---|---|---|
| Decode throughput (tok/s) | 14.3 | 27.4 | **1.91×** |
| Decode latency (ms/tok) | 69.79 | 36.47 | 1.91× faster |
| Acceptance length (tokens/step) | — | 2.53 | — |
| Prefill / TTFT (ms) | 104.7 | 108.5 | — |
| Peak VRAM (GB) | 13.71 | 13.66 | -0.04 |

**Headline: EAGLE3 speculative decoding delivers a 1.91× decode speedup**
(acceptance length 2.53 tokens/verification step).
