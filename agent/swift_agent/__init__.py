"""Swift-VLM-Flow agentic integration.

An observe–reason–act (ORA) agent that wraps the quantized,
speculative-decoding-accelerated VLM behind an OpenAI-compatible chat
endpoint. The agent ingests a visual observation, prompts the VLM for a
chain-of-thought trace, parses structured tool calls out of the
response, executes them, and feeds results back for the next step.

The design decouples perception (the served VLM) from control (this
agent), so the inference backend — vLLM, TensorRT-LLM, or sglang serving
Qwen2.5-VL — can be swapped without touching the agent loop.

Public surface
--------------
- ``VLMBackend`` / ``BackendConfig`` — OpenAI-compatible client.
- ``Agent`` / ``AgentConfig`` — the observe–reason–act loop.
- ``Tool`` / ``ToolRegistry`` — structured tool-call dispatch.
- ``AgentResult`` / ``AgentStep`` — structured run output.
"""

from swift_agent.agent import Agent, AgentConfig, AgentResult, AgentStep
from swift_agent.backend import BackendConfig, VLMBackend
from swift_agent.tools import Tool, ToolCall, ToolRegistry, ToolResult

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResult",
    "AgentStep",
    "VLMBackend",
    "BackendConfig",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
]
