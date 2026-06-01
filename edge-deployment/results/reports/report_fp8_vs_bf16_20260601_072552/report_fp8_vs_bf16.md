# Qwen2.5-VL-7B EAGLE3 Speculative Decoding — fp8 vs bf16 (TensorRT-Edge-LLM, GB10)

LLaVA-Bench-in-the-Wild, 60 samples, greedy. fp8 = fp8 LLM (base + EAGLE3 draft) + fp16 visual, quantized/exported on an 8×B200 node and built into engines on the GB10 (sm_121); bf16 = exported and built entirely on the GB10.

## SD-on (EAGLE3): fp8 vs bf16

| Metric | bf16 SD-on | fp8 SD-on | Δ |
|---|---|---|---|
| Decode throughput (tok/s) | 27.4 | 47.3 | **+72%** |
| Decode latency (ms/tok) | 36.47 | 21.14 | 1.72× faster |
| Acceptance length (tok/step) | 2.53 | 2.56 | draft lossless-ness preserved |
| Prefill / TTFT (ms) | 108.5 | 68.1 | — |
| Vision encoder (ms) | 45.1 | 44.9 | — |
| Peak VRAM (GB) | 13.66 | 7.55 | **-45%** |

**fp8 lifts decode throughput +72% and cuts peak VRAM -45% vs bf16, with acceptance length preserved (2.56 ≈ 2.53) — the EAGLE3 draft stays (near-)lossless under fp8.**

## SD speedup (vs same-precision autoregressive baseline)

| Precision | SD-off (tok/s) | SD-on (tok/s) | Decode speedup |
|---|---|---|---|
| bf16 | 14.7 | 27.4 | **1.86× (measured)** |
| fp8 | 27.0 | 47.3 | **1.75× (measured)** |

<sub>sources: bf16/SD-off=base_20260601_071244_bf16_sameengine.json, bf16/SD-on=spec_20260531_063457.json, fp8/SD-off=base_20260601_061516_fp8.json, fp8/SD-on=spec_20260601_055943_fp8.json</sub>
