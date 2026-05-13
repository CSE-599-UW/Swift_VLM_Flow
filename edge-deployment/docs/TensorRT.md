# TensorRT Engine Conversion — Qwen2-VL-2B

**CSE 599S, University of Washington**

---

## 1. Overview

TensorRT-LLM converts a pretrained HuggingFace model into hardware-specific
CUDA kernels that run significantly faster than the original PyTorch graph.
For Qwen2-VL-2B-Instruct the expected throughput improvement is **1.5×–2.5×**,
consistent with published benchmarks for similarly sized decoder-only
transformers on Ampere/Ada/Blackwell GPUs.

Key optimization mechanisms applied during engine build:

- **Kernel fusion** — attention, layer-norm, and FFN ops are merged into
  single CUDA kernels, eliminating intermediate tensor round-trips to global
  memory.
- **Static shape optimization** — shapes fixed at build time allow the
  compiler to eliminate branch logic and optimize register allocation.
- **Paged KV cache (PagedAttention)** — KV cache is allocated in fixed-size
  pages rather than contiguous buffers, reducing peak VRAM and enabling
  larger effective batch sizes.
- **Hardware-specific CUDA kernel selection** — `trtllm-build` profiles
  available kernels at compile time and selects the fastest variant for the
  target GPU.

The model is split into two independently loaded engines:

| Engine | File | Size | Role |
|--------|------|------|------|
| LLM decoder | `llm/rank0.engine` | ~3.4 GB | Transformer decoder (language model) |
| Vision encoder | `vision/model.engine` | ~1.3 GB | ViT visual token extractor |

At inference time TensorRT-LLM loads both engines and passes visual tokens
from the vision encoder directly into the LLM decoder's multimodal input
slots, without re-entering Python or copying through CPU memory.

---

## 2. Prerequisites

| Item | Requirement |
|------|-------------|
| GPU | 16 GB+ VRAM (tested: NVIDIA RTX 5060 Ti, 16 GB) |
| CUDA driver | 12.8+ |
| System RAM | 30 GB+ (Stage 3 ONNX export peaks at ~20 GB) |
| Swap | 32 GB recommended (see [Known Issues §6.1](#61-oom-killed-during-stage-3)) |
| Docker image | `vlm-bench:latest` (built from `edge-deployment/Dockerfile`) |
| Model weights | `Qwen2-VL-2B-Instruct` at `/workspace/models/` (see `models/README.md`) |

---

## 3. Engine Directory Structure

After a successful build the engine tree looks like this:

```
trt_engines/
└── qwen2vl/
    ├── checkpoint/          # Stage 1 output: TRT-LLM weight format
    │   ├── config.json      #   model architecture + dtype metadata
    │   └── rank0.safetensors#   reformatted weight tensors (fp16)
    │
    ├── llm/                 # Stage 2 output: compiled LLM decoder engine
    │   ├── config.json      #   engine build config (shapes, plugins)
    │   └── rank0.engine     #   compiled CUDA kernel binary (~3.4 GB)
    │
    └── vision/              # Stage 3 output: compiled vision encoder engine
        ├── config.json      #   ONNX → TRT build metadata
        └── model.engine     #   compiled ViT CUDA binary (~1.3 GB)
```

> **Note**: `run.py` hardcodes the subfolder names `llm/` and `vision/`.
> Do not rename them (see [Known Issues §6.3](#63-engine-folder-names-must-be-exactly-llm-and-vision)).

---

## 4. Quick Start

```bash
# Inside container
chmod +x /workspace/scripts/build_trt_engines.sh
/workspace/scripts/build_trt_engines.sh
```

The script runs all three build stages sequentially, applies the fp16 RAM
patch automatically, and finishes with a smoke test. Total wall-clock time:
approximately **10–20 minutes**, depending on GPU and RAM speed.

---

## 5. Manual Step-by-Step

### Stage 1 — HF Checkpoint Conversion

**What it does**: Reads the HuggingFace weight tensors and rewrites them
into TRT-LLM's internal naming and layout conventions. No CUDA compilation
occurs at this stage; it is a pure tensor reformatting pass.

**Command**:
```bash
python3 /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir /workspace/models/Qwen2-VL-2B-Instruct \
  --output_dir /workspace/trt_engines/qwen2vl/checkpoint \
  --dtype float16
```

**Expected output**:
```
checkpoint/
├── config.json
└── rank0.safetensors
```

**Timing**: ~5 seconds

**Verify**:
```bash
ls -lh /workspace/trt_engines/qwen2vl/checkpoint/
```

---

### Stage 2 — LLM Decoder Engine (`trtllm-build`)

**What it does**: Compiles the transformer decoder layers into optimized
CUDA kernels. Key optimizations enabled by the build flags:

- `--gemm_plugin=float16` — uses cuBLAS fp16 GEMM kernels for matrix
  multiplications in attention and FFN layers
- `--gpt_attention_plugin=float16` — enables fused multi-head attention with
  paged KV cache and `remove_input_padding` (eliminates padding tokens from
  computation entirely)
- Shape parameters fix the kernel signatures at compile time:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_batch_size` | 4 | supports small-batch edge inference |
| `max_input_len` | 2048 | text prompt budget |
| `max_seq_len` | 3072 | input + output combined |
| `max_multimodal_len` | 1296 | 4 × 324 visual tokens (504×504 image → 18×18 patch grid × 1 frame) |

**Command**:
```bash
trtllm-build \
  --checkpoint_dir /workspace/trt_engines/qwen2vl/checkpoint \
  --output_dir /workspace/trt_engines/qwen2vl/llm \
  --gemm_plugin=float16 \
  --gpt_attention_plugin=float16 \
  --max_batch_size=4 \
  --max_input_len=2048 \
  --max_seq_len=3072 \
  --max_multimodal_len=1296
```

**Expected output**:
```
llm/
├── config.json
└── rank0.engine   (~3.4 GB)
```

**Timing**: ~30 seconds

**Verify**:
```bash
ls -lh /workspace/trt_engines/qwen2vl/llm/
```

---

### Stage 3 — Vision Encoder Engine (`build_multimodal_engine.py`)

**What it does**: Exports the Qwen2-VL vision transformer (ViT) to ONNX
via `torch.onnx.export()` (graph tracing), then invokes the TensorRT ONNX
parser to compile the ONNX graph into a TRT engine binary. Two sub-phases:

1. **ONNX export** (~8–15 min): traces the ViT forward pass and serializes
   the computation graph. This is the RAM-intensive phase (see Known Issues).
2. **TRT engine build** (~30 sec): TRT optimizes and compiles the ONNX graph
   into hardware-specific CUDA kernels.

Before running Stage 3, the build script patches `multimodal_builder.py` to
load the ViT in fp16 instead of fp32, reducing peak RAM from ~25 GB to ~15 GB
(see [Known Issues §6.2](#62-fp32--fp16-patch-in-multimodal_builderpy)).

**Command**:
```bash
python3 /app/tensorrt_llm/examples/models/core/multimodal/build_multimodal_engine.py \
  --model_type qwen2_vl \
  --model_path /workspace/models/Qwen2-VL-2B-Instruct \
  --output_dir /workspace/trt_engines/qwen2vl/vision \
  --max_batch_size 1
```

**Expected output**:
```
vision/
├── config.json
└── model.engine   (~1.3 GB)
```

**Timing**: 8–15 min (ONNX export) + ~30 sec (TRT build)

**Verify**:
```bash
ls -lh /workspace/trt_engines/qwen2vl/vision/
```

---

### Stage 4 — Smoke Test

**What it does**: Loads both compiled engines and runs a single
image-plus-question inference through the full pipeline end-to-end,
confirming that the engine files are valid and the vision-language bridge
is correctly wired.

**Command**:
```bash
python3 /app/tensorrt_llm/examples/models/core/multimodal/run.py \
  --hf_model_dir /workspace/models/Qwen2-VL-2B-Instruct \
  --engine_dir /workspace/trt_engines/qwen2vl
```

**Expected output**: A coherent natural-language description of a beach scene
(the default test image bundled with TensorRT-LLM multimodal examples).

**Verify**: The output text makes factual sense. Any CUDA error, segfault, or
empty output indicates a failed engine build — re-run the affected stage.

---

## 6. Known Issues

### 6.1 OOM (Killed) During Stage 3

**Symptom**: The Stage 3 process is killed mid-run with no Python traceback,
only `Killed` printed to the terminal.

**Root cause**: `torch.onnx.export()` loads the full ViT model in fp32
(~8 GB) and simultaneously retains all intermediate layer activations in
memory during graph tracing (~12 GB additional), totaling ~20 GB peak RAM
consumption. Systems with less than 20 GB free RAM will trigger the OOM
killer.

**Fix**: Expand swap to 32 GB on the **host machine** before starting the
container. Run these commands on the host (not inside the container):

```bash
sudo swapoff -a
sudo dd if=/dev/zero of=/swapfile bs=1G count=32
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# Verify:
free -h
```

Then restart the container and re-run Stage 3.

---

### 6.2 fp32 → fp16 Patch in `multimodal_builder.py`

**Why needed**: The default `torch_dtype=torch.float32` in
`multimodal_builder.py` causes ~25 GB peak RAM during ONNX export, which
exceeds available memory on 30 GB systems and triggers OOM even with 32 GB
swap.

**The patch** changes the single line:
```python
# Before
torch_dtype=torch.float32,
# After
torch_dtype=torch.float16,
```

**Why it is safe**: ONNX export via `torch.onnx.export()` performs graph
tracing only — it records the sequence of operations and tensor shapes, but
does not execute inference or produce numerical outputs that need to be
accurate. The fp16 change halves the model's RAM footprint during tracing
(~8 GB → ~4 GB) without affecting the final TRT engine precision, because
TensorRT re-derives all kernel precisions independently during engine
compilation.

**Backup**: The build script writes a backup to `multimodal_builder.py.bak`
before applying the patch, allowing easy rollback:
```bash
cp /usr/local/lib/python3.12/dist-packages/tensorrt_llm/tools/multimodal_builder.py.bak \
   /usr/local/lib/python3.12/dist-packages/tensorrt_llm/tools/multimodal_builder.py
```

---

### 6.3 Engine Folder Names Must Be Exactly `llm/` and `vision/`

**Symptom**: `run.py` raises a `FileNotFoundError` or `AssertionError`
referencing a missing engine path.

**Root cause**: `run.py` constructs engine file paths by appending the
hardcoded subdirectory names `llm` and `vision` to `--engine_dir`. There is
no configuration option to override these names.

**Fix**: Always use exactly `llm/` and `vision/` as output subdirectories.
Do not rename them to `llm_engine/`, `vision_encoder/`, or similar. The
build script creates the correct names automatically.

---

## 7. Benchmark Results

The table below will be filled after `run_benchmark_trt.py` is complete.
Baseline (PyTorch fp16) values are from Run ID `20260509_030136` on an
NVIDIA RTX 5060 Ti (16 GB), 50 samples, 3 warmup iterations.

| Metric | PyTorch fp16 | TRT fp16 | TRT fp8 |
|--------|:-----------:|:--------:|:-------:|
| TTFT (ms) | 127.8 | TBD | TBD |
| Throughput (tok/s) | 51.0 | TBD | TBD |
| Peak VRAM (GB) | 4.26 | TBD | TBD |
| VQAv2 Accuracy | 83.4% | TBD | TBD |
| Speedup | 1.0× | TBD | TBD |

*TBD values will be filled after `run_benchmark_trt.py` is complete.*

---

## 8. Related Work

- **LiteVLM** (arXiv:2506.07416): Edge-optimized VLM deployment achieving
  2.5× latency reduction with fp16 and 3.2× with FP8 quantization on
  comparable hardware; motivates the fp8 conversion target in this project.
- **MBQ** (CVPR 2025): Mixed-precision block quantization study for
  Qwen2-VL, providing accuracy reference points for VQAv2 under aggressive
  quantization.
- **TensorRT-LLM documentation**:
  https://nvidia.github.io/TensorRT-LLM/

---

*Swift-VLM-Flow — CSE 599S, University of Washington*
