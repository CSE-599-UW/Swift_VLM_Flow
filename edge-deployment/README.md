# Edge Deployment — Swift-VLM-Flow

**CSE 599S, University of Washington**

Benchmarks Qwen2-VL-2B-Instruct across seven quantization tiers using TensorRT-LLM.
Measures **efficiency** (TTFT, decode latency, VRAM) and **accuracy** (VQAv2, POPE, MME)
for the HuggingFace PyTorch baseline and six TRT engine precisions.

---

## Hardware

- GPU: NVIDIA RTX 5060 Ti (16 GB VRAM) — fp8 and nvfp4 require Ada/Blackwell
- CUDA Driver: 12.8+
- System RAM: 32 GB+ (Stage 3 vision encoder build peaks at ~20 GB)

---

## Quick Start

### Step 1 — Get model weights

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \
  --local-dir ./models/Qwen2-VL-2B-Instruct
```

### Step 2 — Build Docker image (first time, ~10–15 min)

```bash
docker build -t vlm-bench:latest .
```

### Step 3 — Start container

```bash
docker run -it --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/Workspace/Course/CSE599s/Project/Swift_VLM_Flow/edge-deployment/models:/workspace/models \
  vlm-bench:latest bash
```

### Step 4 — Build TRT engines (inside container)

```bash
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant bf16
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant int8
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant int4
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant smoothquant
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant fp8
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant int4_awq
bash /workspace/scripts/build_trt_engines.sh --model qwen2vl_2b --quant nvfp4
```

See [scripts/README.md](scripts/README.md) for build details and the OOM warning for Stage 3.

### Step 5 — Run efficiency benchmark

```bash
cd /workspace/benchmark/efficiency
bash run_efficiency_all.sh
```

### Step 6 — Run accuracy benchmark

```bash
cd /workspace/benchmark/accuracy
bash run_accuracy_all.sh
```

### Step 7 — Generate report

```bash
cd /workspace/benchmark
python3 report.py --latest
# Output: results/reports/report_<timestamp>/report.md + img/*.png
```

---

## Repository Layout

```
edge-deployment/
├── benchmark/
│   ├── efficiency/     # latency & VRAM benchmark
│   ├── accuracy/       # VQAv2 / POPE / MME benchmark
│   ├── report.py       # chart + Markdown report generator
│   └── README.md
├── scripts/
│   ├── build_trt_engines.sh   # TRT engine builder (all quant modes)
│   └── README.md
├── models/             # HF model weights (gitignored)
├── results/            # benchmark outputs (gitignored except reports/)
│   ├── efficiency/
│   ├── accuracy/
│   └── reports/        # tracked in git
└── Dockerfile
```

---

## Quantization Tiers

| Tier | Precision | Build Pipeline |
|---|---|---|
| PyTorch BF16 | W16A16 | HF baseline, no engine needed |
| TRT BF16 | W16A16 | Pipeline A |
| TRT INT8 | W8A16 | Pipeline A |
| TRT INT4 | W4A16 | Pipeline A |
| TRT SmoothQuant | W8A8 | Pipeline A |
| TRT FP8 | W8A8 | Pipeline B (ModelOpt, Ada/Hopper/Blackwell) |
| TRT INT4-AWQ | W4A16 | Pipeline B (ModelOpt) |
| TRT NVFP4 | W4A8 | Pipeline B (ModelOpt, Blackwell only) |

---

## Metrics

| Metric | Description |
|---|---|
| `ttft_ms` | Time to First Token — prefill cost |
| `decode_latency_ms_per_tok` | Decode time per output token |
| `static_vram_gb` | VRAM after model/engine load |
| `dynamic_vram_gb` | Per-inference VRAM above static baseline |
| VQAv2 accuracy | Soft-match against 10 human annotations (500 samples) |
| POPE avg F1 | Object hallucination across 3 splits × 500 samples |
| MME total score | Perception (10 tasks) + Cognition (4 tasks) pair scoring |

---

## Related Papers

- LiteVLM (arXiv:2506.07416) — edge VLM, FP8, 2.5×–3.2× speedup
- MBQ CVPR 2025 — Qwen2-VL quantization with lmms-eval
- GRACE (arXiv:2601.22709) — Qwen2-VL-2B INT4 baseline numbers
- Edge Reliability Gap (arXiv:2603.26769) — VQAv2 on RTX hardware

---

*Swift-VLM-Flow — CSE 599S, University of Washington*
