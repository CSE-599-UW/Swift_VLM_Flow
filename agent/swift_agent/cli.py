"""CLI for the Swift-VLM-Flow agent.

Run an observe–reason–act episode against a served Qwen2.5-VL endpoint::

    python -m swift_agent.cli \\
        --base-url http://127.0.0.1:8000/v1 \\
        --model Qwen2.5-VL-7B-Instruct \\
        --image ../edge-deployment/test_assets/sample.jpg \\
        --prompt "What should the robot do to pick up the mug?"

Requires a running OpenAI-compatible server (vLLM / TRT-LLM / sglang).
Without one, see ``demo_offline`` for a mock-backed dry run that needs
no GPU.
"""

from __future__ import annotations

import argparse
import json

from swift_agent.agent import Agent, AgentConfig
from swift_agent.backend import BackendConfig, VLMBackend
from swift_agent.example_tools import default_registry


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Swift-VLM-Flow observe–reason–act agent")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                   help="OpenAI-compatible server root (…/v1)")
    p.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    p.add_argument("--prompt", required=True, help="Task / question text")
    p.add_argument("--image", default=None, help="Optional image path for the first observation")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--json", action="store_true", help="Emit the full result as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    backend = VLMBackend(BackendConfig(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ))
    agent = Agent(
        backend=backend,
        tools=default_registry(),
        config=AgentConfig(max_steps=args.max_steps),
    )
    result = agent.run(prompt=args.prompt, image_path=args.image)

    if args.json:
        print(json.dumps({
            "final_answer": result.final_answer,
            "stopped_reason": result.stopped_reason,
            "num_steps": len(result.steps),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "steps": [
                {
                    "index": s.index,
                    "is_final": s.is_final,
                    "tool": s.tool_result.tool if s.tool_result else None,
                    "tool_ok": s.tool_result.ok if s.tool_result else None,
                }
                for s in result.steps
            ],
        }, indent=2))
    else:
        for s in result.steps:
            tag = "FINAL" if s.is_final else (
                f"tool:{s.tool_result.tool}" if s.tool_result else "think"
            )
            print(f"[step {s.index}] ({tag})")
            print(s.reasoning.strip())
            if s.tool_result is not None:
                print("  ->", s.tool_result.as_observation())
            print()
        print("=" * 60)
        print(f"final answer ({result.stopped_reason}): {result.final_answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
