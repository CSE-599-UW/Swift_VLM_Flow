# Agentic Integration — Swift-VLM-Flow

**CSE 599S, University of Washington** · integration track (§3.3 of the report)

The two acceleration tracks (quantization, speculative decoding) converge
into an end-to-end VLM **agent** for robotic and small-batch inference. The
agent wraps the quantized, speculative-decoding-accelerated VLM in an
**observe → reason → act** loop:

1. **Observe** — ingest a visual frame (camera image / screenshot) + text context.
2. **Reason** — prompt the VLM for a chain-of-thought trace.
3. **Act** — parse structured tool calls from the reply and execute them,
   feeding results back as the next observation until a final answer.

Perception (the served VLM) is **decoupled** from control (this loop), so the
inference backend — vLLM, TensorRT-LLM, or sglang serving Qwen2.5-VL — can be
swapped without touching the agent. The only assumption is an
OpenAI-compatible `/v1/chat/completions` endpoint.

---

## Layout

```
agent/
  swift_agent/
    backend.py        # OpenAI-compatible chat client (image+text content parts)
    tools.py          # tool-call protocol: parse + registry + dispatch
    agent.py          # the observe–reason–act loop
    example_tools.py  # move / grasp / report demo tools
    cli.py            # `python -m swift_agent.cli ...`
  tests/              # 32 unit tests, mocked backend (no GPU/server)
  demo_offline.py     # scripted end-to-end loop, runs anywhere
  pyproject.toml
```

## Quick start (offline — no GPU)

```bash
cd agent
PYTHONPATH=. python demo_offline.py     # scripted observe→reason→act episode
PYTHONPATH=. python -m pytest tests/    # 32 tests
```

## Live run (against a served VLM)

Start any OpenAI-compatible server for `Qwen2.5-VL-7B-Instruct` (the team's
benchmark target), e.g. with vLLM:

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000
```

Then drive the agent:

```bash
cd agent
PYTHONPATH=. python -m swift_agent.cli \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen2.5-VL-7B-Instruct \
  --image ../edge-deployment/test_assets/sample.jpg \
  --prompt "What should the robot do to pick up the mug?"
```

To point the agent at the **optimized** pipeline, serve the FP8 +
EAGLE3 engine from `edge-deployment/` behind the same OpenAI API and pass
its `--base-url` — no agent changes needed.

## Tool-call protocol

The agent is model-agnostic: instead of a vendor function-calling API, the
VLM is instructed to end its reply with a fenced block:

````
```tool
{"tool": "move", "args": {"direction": "left", "distance": 0.5}}
```
````

`parse_tool_call` extracts the first block; a reply with **no** block is the
final answer (loop terminates). Malformed JSON is fed back to the model as a
`[parse-error]` observation for self-repair, and tool exceptions become
`[tool:…] ERROR` observations rather than crashing the loop.

## Design notes

- **Backend seam.** `VLMBackend._post` is the single network call; tests
  inject a `post_fn` / subclass `chat`, so the whole loop is testable with
  no GPU or server (see `tests/`).
- **Image convention.** Multimodal user turns send
  `[{type:image_url}, {type:text}]` content parts, mirroring the team's
  speculative benchmark `io_builder.py`.
- **Robustness.** The loop is bounded by `max_steps`; tool errors and parse
  errors are recoverable (surfaced as observations), so a misbehaving model
  degrades gracefully instead of crashing.
