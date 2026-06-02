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
## Mixed-precision ablation (base × draft, SD-on)

Same eagle engine dir with the base and draft engines swapped across precisions (symlinked, no rebuild). Acceptance stays ~2.5 in every combo, so the draft predicts equally well regardless — differences are pure runtime overhead.

Decode throughput (tok/s):

| base \ draft | bf16 | fp8 |
|---|---|---|
| **bf16** | 27.4 | 15.4 |
| **fp8** | 23.0 | 47.3 |

| base | draft | tok/s | acceptance | VRAM (GB) | vs same-base autoregressive |
|---|---|---|---|---|---|
| bf16 | bf16 | 27.4 | 2.53 | 13.66 | 1.86× |
| bf16 | fp8 | 15.4 | 2.52 | 13.69 | 1.04× |
| fp8 | bf16 | 23.0 | 2.54 | 7.54 | 0.85× |
| fp8 | fp8 | 47.3 | 2.56 | 7.55 | 1.75× |

**Matched precision is required.** Both mixes fall *below pure bf16*, and fp8-base + bf16-draft is even slower than decoding the fp8 base autoregressively (net-negative SD). EAGLE feeds the base's hidden states (dim 10752) to the draft every step (1 verify + ~6 draft forwards/token); a precision mismatch forces a dtype conversion on each hop that swamps any per-engine gain. Peak VRAM tracks the **base** precision. → fp8 must be applied to **both** base and draft to get the win.


<sub>sources: bf16/SD-off=base_20260601_071244_bf16_sameengine.json, bf16/SD-on=spec_20260531_063457.json, fp8/SD-off=base_20260601_061516_fp8.json, fp8/SD-on=spec_20260601_055943_fp8.json, mix_b16base_fp8draft/SD-on=spec_20260602_021301_mixA_b16base_fp8draft.json, mix_fp8base_bf16draft/SD-on=spec_20260602_022540_mixB_fp8base_bf16draft.json</sub>
