"""Observe–reason–act (ORA) agent over a served VLM.

The agent implements the integration layer from the Swift-VLM-Flow
report (§3.3): it wraps the quantized, speculative-decoding-accelerated
VLM in an execution loop that

  1. **Observes** — ingests a visual frame + text context,
  2. **Reasons** — prompts the VLM for a chain-of-thought trace, and
  3. **Acts** — parses structured tool calls and executes them,

feeding tool results back as observations until the VLM emits a final
answer (a reply with no tool block) or the step budget is exhausted.

Perception (the served VLM) is decoupled from control (this loop), so
the backend can be swapped without changing the agent. The only model
assumption is an OpenAI-compatible chat endpoint (see ``backend.py``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from swift_agent.backend import VLMBackend, build_user_content
from swift_agent.tools import ToolRegistry, ToolResult, parse_tool_call

SYSTEM_PROMPT = """You are a vision-language agent operating an observe–reason–act loop.

At each step you receive an observation (an image and/or text). Think
step by step about what you see, then EITHER:

  (a) call ONE tool by ending your reply with a fenced block:
      ```tool
      {{"tool": "<name>", "args": {{...}}}}
      ```
  (b) give your FINAL ANSWER as plain text with no tool block.

Only the FIRST tool block in your reply is executed. After a tool runs,
you will receive its result as the next observation. When you have
enough information, stop calling tools and give the final answer.

Available tools:
{tool_descriptions}
"""


@dataclasses.dataclass
class AgentConfig:
    """Controls the ORA loop."""

    max_steps: int = 6
    # When True, the agent stops the moment a reply has no tool call.
    # (Always True today; kept explicit for future "must-call" modes.)
    stop_on_final_answer: bool = True


@dataclasses.dataclass
class AgentStep:
    """One iteration of the loop, for inspection / logging / eval."""

    index: int
    reasoning: str               # the VLM's full reply (CoT + any tool block)
    tool_result: ToolResult | None = None
    is_final: bool = False


@dataclasses.dataclass
class AgentResult:
    """The full outcome of a ``run``."""

    final_answer: str
    steps: list[AgentStep]
    stopped_reason: str           # "final_answer" | "max_steps"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Agent:
    """Drives the observe–reason–act loop over a ``VLMBackend``."""

    def __init__(
        self,
        backend: VLMBackend,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        self.backend = backend
        self.tools = tools or ToolRegistry()
        self.config = config or AgentConfig()

    def _system_message(self) -> dict[str, Any]:
        content = SYSTEM_PROMPT.format(tool_descriptions=self.tools.describe())
        return {"role": "system", "content": content}

    def run(
        self,
        prompt: str,
        image_path: str | Path | None = None,
    ) -> AgentResult:
        """Run the loop for one task.

        Args:
            prompt: the task / question text (the initial observation).
            image_path: optional image to attach to the first user turn.
                Only the first observation carries the image — subsequent
                turns are tool-result text, matching the observe-once /
                reason-act-repeat structure.

        Returns:
            ``AgentResult`` with the final answer and per-step trace.
        """
        messages: list[dict[str, Any]] = [self._system_message()]
        messages.append(
            {"role": "user", "content": build_user_content(prompt, image_path)}
        )

        steps: list[AgentStep] = []
        prompt_tokens = completion_tokens = 0

        for i in range(self.config.max_steps):
            resp = self.backend.chat(messages)
            prompt_tokens += resp.prompt_tokens
            completion_tokens += resp.completion_tokens
            reply = resp.text
            messages.append({"role": "assistant", "content": reply})

            # Parse the act step. A parse error is fed back to the model
            # as an observation so it can self-correct rather than crash.
            try:
                call = parse_tool_call(reply)
            except ValueError as e:
                steps.append(AgentStep(index=i, reasoning=reply))
                messages.append({
                    "role": "user",
                    "content": f"[parse-error] {e}. Re-emit a valid tool block or give a final answer.",
                })
                continue

            if call is None:
                # No tool call -> final answer; loop terminates.
                steps.append(AgentStep(index=i, reasoning=reply, is_final=True))
                return AgentResult(
                    final_answer=reply.strip(),
                    steps=steps,
                    stopped_reason="final_answer",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            result = self.tools.dispatch(call)
            steps.append(AgentStep(index=i, reasoning=reply, tool_result=result))
            messages.append({
                "role": "user",
                "content": result.as_observation(),
            })

        # Step budget exhausted without a final answer. Surface the last
        # reply as the answer so callers always get something usable.
        last_reply = steps[-1].reasoning.strip() if steps else ""
        return AgentResult(
            final_answer=last_reply,
            steps=steps,
            stopped_reason="max_steps",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


__all__ = ["Agent", "AgentConfig", "AgentStep", "AgentResult", "SYSTEM_PROMPT"]
