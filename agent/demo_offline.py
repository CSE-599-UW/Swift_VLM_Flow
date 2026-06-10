"""Offline demo of the observe–reason–act loop — no GPU / server needed.

Drives the Agent with a scripted mock backend so the loop, tool
dispatch, and final-answer extraction can be seen end-to-end on any
machine::

    cd agent && PYTHONPATH=. python demo_offline.py

For a real run against a served Qwen2.5-VL endpoint, use the CLI:

    PYTHONPATH=. python -m swift_agent.cli --base-url http://127.0.0.1:8000/v1 \\
        --prompt "What should the robot do?" --image ../edge-deployment/test_assets/sample.jpg
"""

from __future__ import annotations

from swift_agent.agent import Agent, AgentConfig
from swift_agent.backend import ChatResponse, VLMBackend
from swift_agent.example_tools import default_registry


class _ScriptedBackend(VLMBackend):
    """Returns canned VLM replies so the demo runs without a server."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self._replies = list(replies)

    def chat(self, messages):  # type: ignore[override]
        reply = self._replies.pop(0) if self._replies else "Final answer: done."
        return ChatResponse(text=reply, prompt_tokens=8, completion_tokens=4)


def main() -> None:
    # A two-step robotic episode: observe -> move -> observe result -> answer.
    backend = _ScriptedBackend([
        "I see a mug on the left edge of the frame. I should move left to center it.\n"
        '```tool\n{"tool": "move", "args": {"direction": "left", "distance": 0.3}}\n```',
        "The mug is now centered and within reach. I will grasp it.\n"
        '```tool\n{"tool": "grasp", "args": {"target": "red mug"}}\n```',
        "Task complete: the red mug has been grasped. Final answer: grasped the red mug.",
    ])
    agent = Agent(backend, default_registry(), AgentConfig(max_steps=6))
    result = agent.run(
        prompt="Pick up the red mug.",
        image_path=None,  # mock backend ignores the image
    )

    for step in result.steps:
        tag = "FINAL" if step.is_final else (
            f"tool:{step.tool_result.tool}" if step.tool_result else "think"
        )
        print(f"\n--- step {step.index} ({tag}) ---")
        print(step.reasoning.strip())
        if step.tool_result is not None:
            print("  ->", step.tool_result.as_observation())

    print("\n" + "=" * 60)
    print(f"stopped: {result.stopped_reason}")
    print(f"final answer: {result.final_answer}")
    print(f"tokens: prompt={result.prompt_tokens} completion={result.completion_tokens}")


if __name__ == "__main__":
    main()
