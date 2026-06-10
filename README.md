# Swift-VLM-Flow

**Accelerating Vision-Language Model Inference via Quantization, Speculative
Decoding, and Agentic Integration**

CSE 599S, University of Washington · Jack Xiao, Steven Gao, Hongyu Mao,
Chi-Lung Ma, Noah Meng

A systematic study of inference-acceleration techniques for Vision-Language
Models (VLMs) targeting edge and robotic deployment. We evaluate TensorRT-LLM
quantization configurations on Qwen2-VL-2B-Instruct, integrate EAGLE3
speculative decoding with Qwen2.5-VL-7B-Instruct, and wrap the optimized model
in an observe–reason–act agent.

📄 **Final report:** [`edge-deployment/results/reports/final_report.md`](edge-deployment/results/reports/final_report.md)

---

## Repository map

The project has two acceleration tracks and one integration layer; each lives
in its own top-level component.

| Component | Direction | What's here |
|---|---|---|
| [`edge-deployment/`](edge-deployment/) | **Quantization** (Direction 1) + **Speculative Decoding** (Direction 2) | TensorRT-LLM engine conversion (FP16/FP8/INT4-AWQ/NVFP4/SmoothQuant) + EAGLE3 speculative-decoding benchmarks on Qwen2.5-VL-7B. Efficiency + accuracy + spec-decode harnesses, results, and the final report. |
| [`agent/`](agent/) | **Agentic Integration** (§3.3) | Observe–reason–act agent that wraps the optimized VLM behind an OpenAI-compatible endpoint. Backend client, tool-call protocol, the ORA loop, 32 unit tests, offline demo. |

External companion repos (separate course sub-projects):

- **BitNet 1.58-bit ternary VLM** — <https://github.com/CSE-599-UW/ternaryvlm>

---

## Key findings (see the report for full tables)

1. **SmoothQuant (W8A8)** — best accuracy–efficiency trade-off: 2.13× decode
   speedup, +0.5 pp VQAv2 vs FP16.
2. **NVFP4 (W4A8)** — highest raw decode throughput: 2.42× speedup.
3. **EAGLE3 speculative decoding** — 1.89× decode speedup on BF16
   Qwen2.5-VL-7B; 46.9 tok/s (+69%) at 7.54 GB VRAM (−45%) with FP8.
4. **FP8 KV-cache breaks VLMs** — the standard text-LLM `--kv_cache_dtype fp8`
   recipe causes −8.8 pp VQAv2 on Qwen2-VL (visual tokens exceed FP8's
   dynamic range). Fix: FP8 weights + BF16 KV cache.

The combined quantization + speculative decoding result, wrapped in the
agent's observe–reason–act loop, is a practical path to real-time VLM
inference on constrained hardware.

---

## Quick start

**Quantization / speculative-decoding benchmarks** — see
[`edge-deployment/README.md`](edge-deployment/README.md) (Docker + TensorRT-LLM,
needs an NVIDIA GPU).

**Agent** (runs offline, no GPU needed):

```bash
cd agent
PYTHONPATH=. python demo_offline.py      # scripted observe→reason→act episode
PYTHONPATH=. python -m pytest tests/     # 32 unit tests
```

To run the agent against a live served VLM, see
[`agent/README.md`](agent/README.md).

---

## Contributing

Create your own branch to develop, then open a pull request to merge into
`main`.
