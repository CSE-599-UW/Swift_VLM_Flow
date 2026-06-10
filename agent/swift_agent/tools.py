"""Structured tool calls for the observe–reason–act agent.

The agent's "act" step parses the VLM's response for tool calls and
dispatches them. We use an explicit, model-agnostic JSON protocol rather
than a vendor function-calling API so the same parser works across any
served backend (vLLM / TRT-LLM / sglang) and any Qwen-VL precision tier.

Protocol
--------
The VLM is instructed (see ``agent.SYSTEM_PROMPT``) to end its reply with
either a final answer or a single fenced tool-call block::

    ```tool
    {"tool": "move", "args": {"direction": "left", "distance": 0.5}}
    ```

``parse_tool_call`` extracts the first such block. A reply with no block
is treated as a final answer (the loop terminates).
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Callable


@dataclasses.dataclass
class ToolCall:
    """A parsed request to invoke a tool."""

    tool: str
    args: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ToolResult:
    """The outcome of executing a tool call."""

    tool: str
    ok: bool
    output: Any = None
    error: str | None = None

    def as_observation(self) -> str:
        """Render the result as text to feed back into the next ORA step."""
        if self.ok:
            return f"[tool:{self.tool}] OK -> {self.output}"
        return f"[tool:{self.tool}] ERROR -> {self.error}"


@dataclasses.dataclass
class Tool:
    """A named, callable tool with a short description for the prompt.

    ``fn`` takes the parsed ``args`` dict and returns any JSON-able
    output. Exceptions raised inside ``fn`` are caught by the registry
    and surfaced as a failed ``ToolResult`` (the agent can then reason
    about the error rather than crashing).
    """

    name: str
    description: str
    fn: Callable[[dict[str, Any]], Any]


# A fenced ```tool ... ``` block. DOTALL so the JSON can span lines;
# non-greedy so we grab the first block, not everything to the last fence.
_TOOL_BLOCK_RE = re.compile(r"```tool\s*(\{.*?\})\s*```", re.DOTALL)
# Fallback: a bare ``{"tool": ...}`` object on its own (some models drop
# the fence). Anchored to a line that starts with ``{"tool"``.
_BARE_TOOL_RE = re.compile(r'(\{\s*"tool"\s*:\s*.*?\})\s*$', re.DOTALL)


def parse_tool_call(text: str) -> ToolCall | None:
    """Extract the first tool call from a VLM reply, or None.

    Returns None when the reply contains no tool block — the agent
    interprets that as a final answer and stops. Malformed JSON inside a
    block raises ``ValueError`` so the caller can feed the parse error
    back to the model as an observation (self-repair).
    """
    match = _TOOL_BLOCK_RE.search(text)
    if match is None:
        match = _BARE_TOOL_RE.search(text.strip())
    if match is None:
        return None

    blob = match.group(1)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed tool-call JSON: {blob!r} ({e})") from e

    if not isinstance(data, dict) or "tool" not in data:
        raise ValueError(f"Tool call missing 'tool' key: {data!r}")
    args = data.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"Tool 'args' must be an object, got {type(args).__name__}")
    return ToolCall(tool=str(data["tool"]), args=args)


class ToolRegistry:
    """Holds the available tools and dispatches calls to them."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        """Human-readable tool list for injection into the system prompt."""
        if not self._tools:
            return "(no tools available)"
        return "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools.values()
        )

    def dispatch(self, call: ToolCall) -> ToolResult:
        """Run a tool call, capturing errors as a failed result."""
        tool = self._tools.get(call.tool)
        if tool is None:
            return ToolResult(
                tool=call.tool, ok=False,
                error=f"unknown tool {call.tool!r}; available: {self.names()}",
            )
        try:
            output = tool.fn(call.args)
        except Exception as e:  # noqa: BLE001 — surface any tool failure to the model
            return ToolResult(tool=call.tool, ok=False, error=f"{type(e).__name__}: {e}")
        return ToolResult(tool=call.tool, ok=True, output=output)


__all__ = [
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "parse_tool_call",
]
