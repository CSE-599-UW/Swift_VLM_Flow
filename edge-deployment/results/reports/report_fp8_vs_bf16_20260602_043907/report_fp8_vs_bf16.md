# Qwen2.5-VL-7B EAGLE3 Speculative Decoding — fp8 vs bf16 (TensorRT-Edge-LLM, GB10)

LLaVA-Bench-in-the-Wild, 60 samples, greedy. fp8 = fp8 LLM (base + EAGLE3 draft) + fp16 visual, quantized/exported on an 8×B200 node and built into engines on the GB10 (sm_121); bf16 = exported and built entirely on the GB10.

## SD-on (EAGLE3): fp8 vs bf16

| Metric | bf16 SD-on | fp8 SD-on | Δ |
|---|---|---|---|
| Decode throughput (tok/s) | 27.7 | 46.9 | **+70%** |
| Decode latency (ms/tok) | 36.15 | 21.31 | 1.70× faster |
| Acceptance length (tok/step) | 2.53 | 2.56 | draft lossless-ness preserved |
| Prefill / TTFT (ms) | 108.2 | 66.0 | — |
| Vision encoder (ms) | 45.0 | 44.9 | — |
| Peak VRAM (GB) | 13.69 | 7.54 | **-45%** |

**fp8 lifts decode throughput +70% and cuts peak VRAM -45% vs bf16, with acceptance length preserved (2.56 ≈ 2.53) — the EAGLE3 draft stays (near-)lossless under fp8.**

## SD speedup (vs same-precision autoregressive baseline)

| Precision | SD-off (tok/s) | SD-on (tok/s) | Decode speedup |
|---|---|---|---|
| bf16 | 14.7 | 27.7 | **1.89× (measured)** |
| fp8 | 27.3 | 46.9 | **1.72× (measured)** |
## Mixed-precision ablation (base × draft, SD-on)

Same eagle engine dir with the base and draft engines swapped across precisions (symlinked, no rebuild). All four combos run fine (acceptance ~2.5 throughout); the throughput differences isolate where the fp8 win comes from. (All six runs measured back-to-back with the GPU verified vacant — vision-encoder time ~45 ms in every run.)

Decode throughput (tok/s):

| base \ draft | bf16 | fp8 |
|---|---|---|
| **bf16** | 27.7 | 28.6 |
| **fp8** | 42.8 | 46.9 |

| base | draft | tok/s | acceptance | VRAM (GB) | vs same-base autoregressive |
|---|---|---|---|---|---|
| bf16 | bf16 | 27.7 | 2.53 | 13.69 | 1.89× |
| bf16 | fp8 | 28.6 | 2.52 | 13.69 | 1.95× |
| fp8 | bf16 | 42.8 | 2.54 | 7.54 | 1.56× |
| fp8 | fp8 | 46.9 | 2.56 | 7.54 | 1.72× |

**Throughput is governed by the base precision** — the base runs the expensive tree-verification forward every step. An fp8 base (43–47 tok/s) is ~70% faster than a bf16 base (28–29 tok/s) regardless of draft. The **draft precision is second-order**: with an fp8 base a bf16 draft costs ~9% (42.8 vs 46.9 tok/s); with a bf16 base it's within noise (27.7 vs 28.6). Acceptance stays ~2.5 throughout and peak VRAM tracks the **base** precision (fp8 base → ~7.5 GB). So fp8 on the base captures most of the win — quantizing the draft too adds only a few percent, and mixing is safe.


<sub>sources: bf16/SD-off=base_20260602_040129_bf16_clean.json, bf16/SD-on=spec_20260602_035420_bf16_clean.json, fp8/SD-off=base_20260602_041825_fp8_clean.json, fp8/SD-on=spec_20260602_041400_fp8_clean.json, mix_b16base_fp8draft/SD-on=spec_20260602_042518_mixA_clean.json, mix_fp8base_bf16draft/SD-on=spec_20260602_043214_mixB_clean.json</sub>
