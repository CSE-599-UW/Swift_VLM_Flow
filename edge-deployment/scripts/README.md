# Build TensorRT Engine

## Usage

Inside the container:

```bash
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant bf16
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant int8
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant int4
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant smoothquant
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant fp8 --no_kv_fp8
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant int4_awq
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant nvfp4
```

### Quantization modes

**Pipeline A** — no calibration data needed (`convert_checkpoint.py`)

| `--quant` | Precision | Notes |
|---|---|---|
| `bf16` | W16A16 | BFloat16 baseline |
| `int8` | W8A16 | INT8 weight-only |
| `int4` | W4A16 | INT4 weight-only |
| `smoothquant` | W8A8 | Weight + activation, α=0.5, per-token/per-channel |

**Pipeline B** — calibration via ModelOpt (`quantize.py`)

| `--quant` | Precision | Notes |
|---|---|---|
| `fp8` | W8A8 | Requires Ada/Hopper/Blackwell (RTX 5060 Ti ✅) — always use `--no_kv_fp8` |
| `int4_awq` | W4A16 | AWQ, group size 128, calib 32 samples |
| `nvfp4` | W4A8 | Blackwell only (RTX 5060 Ti ✅) |

---

## FP8 Note

`--kv_cache_dtype fp8` (the standard text-LLM FP8 recipe) causes severe accuracy regression on Qwen2-VL (VQAv2 −8.8 pp, MME −189). Visual token KV distributions exceed FP8 E4M3's dynamic range (±448), introducing rounding errors in attention score computation. **Always use `--no_kv_fp8`**, which keeps FP8 weights/activations but reverts the KV cache to BF16. Accuracy fully recovers; dynamic VRAM increases by ~0.6 GB. See §5.1 of `results/reports/final_report.md` for ablation details.

## FP8 Ablation Flags

These flags are only valid with `--quant fp8` and write to separate engine directories so existing results are never overwritten.

| Flag | Stages re-run | Engine dir | Purpose |
|---|---|---|---|
| `--no_kv_fp8` | 1 + 2 (Stage 3 symlinked) | `qwen2vl_2b_fp8_no_kv_fp8` | **Recommended** — BF16 KV cache, full accuracy |
| `--no_fp8_fmha` | 2 only (checkpoint reused) | `qwen2vl_2b_fp8_no_fmha` | Ablation — rules out FP8 FMHA kernel |

---

## Stage 1 — Checkpoint Conversion

Converts the HF checkpoint to TRT-LLM format and applies quantization (PTQ).

- Pipeline A: `convert_checkpoint.py` — no calibration data needed
- Pipeline B: `quantize.py` (ModelOpt) — calibration size 512 (fp8/nvfp4) or 32 (int4_awq)

## Stage 2 — LLM Decoder Engine

Builds the `rank0.engine` with `trtllm-build`. Active sequence-length preset:

| Parameter | Value |
|---|---|
| `max_batch_size` | 1 |
| `max_multimodal_len` | 1536 (≥ visual tokens per image) |
| `max_input_len` | 2048 (visual + text tokens) |
| `max_seq_len` | 2560 (`max_input_len` + output buffer) |
| `max_num_tokens` | 2560 |

Other presets available in the script (uncomment to use):

| Preset | `max_input_len` | `max_seq_len` | `max_multimodal_len` |
|---|---|---|---|
| Small  | 1440 | 2048 | 1024 |
| **Default** | **2048** | **2560** | **1536** |
| Medium | 2560 | 3072 | 2048 |
| Large  | 4096 | 4608 | 4096 |

Token flow:
```
image (H×W)
  → 14×14 patches → ViT → 2×2 merge → visual tokens (raw_patches / 4)
  → concat text tokens  → LLM decoder → output tokens
```

## Stage 3 — Vision Encoder Engine

> [!CAUTION] **~30 GB system RAM required for ONNX export**
>
> Add 32 GB swap before running:
> ```bash
> sudo swapoff -a
> sudo dd if=/dev/zero of=/swapfile bs=1G count=32
> sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
> ```

Patches `multimodal_builder.py` to set `torch_dtype=bfloat16` if needed (backup saved as `*.bak`). Takes 8–15 min.

## Smoke Test

Runs one VQAv2 validation sample end-to-end:
```
[Q] Describe what you see in this image in one sentence.
[A] A skateboarder in a green hat is performing a trick over a picnic table, with a crowd watching.
Generated 19 tokens
```

---

## Output

```
/workspace/trt_engines/qwen2vl_2b_<quant>/
├── checkpoint/
├── llm/rank0.engine
└── vision/model.engine
```

**Reference sizes and VRAM (BF16, Qwen2-VL-2B-Instruct):**
```
LLM decoder:    3.4 GB
Vision encoder: 1.3 GB
Total:          4.7 GB

Vision engine:  1282 MiB (GPU)
LLM engine:     3414 MiB (GPU)
KV cache:       6.84 GiB
Execution ctx:   131 MiB
─────────────────────────────
Total:         ~11.7 GiB / 15 GiB
```
