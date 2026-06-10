"""Tests for the observe–reason–act loop (scripted mock backend)."""

from __future__ import annotations

from swift_agent.agent import Agent, AgentConfig
from swift_agent.backend import ChatResponse, VLMBackend
from swift_agent.example_tools import default_registry
from swift_agent.tools import Tool, ToolRegistry


class ScriptedBackend(VLMBackend):
    """A VLMBackend that returns a fixed list of replies in order.

    Records every ``messages`` list it was called with so tests can
    assert on what the loop fed back (observations, tool results).
    """

    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages):  # type: ignore[override]
        self.calls.append([dict(m) for m in messages])
        reply = self._replies.pop(0) if self._replies else "done"
        return ChatResponse(text=reply, prompt_tokens=5, completion_tokens=2)


def test_single_step_final_answer() -> None:
    """No tool block on the first reply -> immediate final answer."""
    backend = ScriptedBackend(["The image shows a red mug."])
    agent = Agent(backend, default_registry())
    result = agent.run("What is in the image?")
    assert result.stopped_reason == "final_answer"
    assert result.final_answer == "The image shows a red mug."
    assert len(result.steps) == 1 and result.steps[0].is_final


def test_tool_call_then_final_answer() -> None:
    """First reply calls a tool; second gives the final answer."""
    backend = ScriptedBackend([
        'Looking left.\n```tool\n{"tool": "move", "args": {"direction": "left", "distance": 0.2}}\n```',
        "Now I can see the mug. Final answer: a mug.",
    ])
    agent = Agent(backend, default_registry())
    result = agent.run("Find the mug", image_path=None)

    assert result.stopped_reason == "final_answer"
    assert result.final_answer == "Now I can see the mug. Final answer: a mug."
    assert len(result.steps) == 2
    # step 0 ran the move tool successfully
    assert result.steps[0].tool_result.tool == "move"
    assert result.steps[0].tool_result.ok
    # the tool result was fed back as the next observation
    last_obs = backend.calls[1][-1]
    assert last_obs["role"] == "user"
    assert "[tool:move] OK" in last_obs["content"]


def test_max_steps_cap_when_model_never_stops() -> None:
    """A model that always calls a tool is bounded by max_steps."""
    forever = '```tool\n{"tool": "report", "args": {"text": "again"}}\n```'
    backend = ScriptedBackend([forever] * 10)
    agent = Agent(backend, default_registry(), AgentConfig(max_steps=3))
    result = agent.run("loop forever")
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3
    assert backend.calls.__len__() == 3  # exactly max_steps backend calls


def test_failed_tool_result_fed_back_and_loop_continues() -> None:
    """A tool error becomes an observation; the agent can recover."""
    backend = ScriptedBackend([
        '```tool\n{"tool": "move", "args": {"direction": "sideways"}}\n```',  # invalid
        "Okay, that direction was invalid. Final answer: cannot move sideways.",
    ])
    agent = Agent(backend, default_registry())
    result = agent.run("move sideways")
    assert result.stopped_reason == "final_answer"
    assert not result.steps[0].tool_result.ok
    fed_back = backend.calls[1][-1]["content"]
    assert "[tool:move] ERROR" in fed_back


def test_parse_error_fed_back_for_self_repair() -> None:
    """Malformed tool JSON -> parse-error observation, loop continues."""
    backend = ScriptedBackend([
        '```tool\n{"tool": "move", "args": {oops}}\n```',  # bad JSON
        "Sorry. Final answer: done.",
    ])
    agent = Agent(backend, default_registry())
    result = agent.run("trigger parse error")
    assert result.stopped_reason == "final_answer"
    # step 0 had no tool_result (parse failed before dispatch)
    assert result.steps[0].tool_result is None
    repair_msg = backend.calls[1][-1]["content"]
    assert "[parse-error]" in repair_msg


def test_system_prompt_includes_tool_descriptions() -> None:
    backend = ScriptedBackend(["final"])
    agent = Agent(backend, default_registry())
    agent.run("hi")
    system_msg = backend.calls[0][0]
    assert system_msg["role"] == "system"
    assert "move:" in system_msg["content"]
    assert "observe–reason–act" in system_msg["content"]


def test_image_attached_only_to_first_user_turn(tmp_path) -> None:
    img = tmp_path / "f.png"
    img.write_bytes(b"\x89PNG fake")
    backend = ScriptedBackend([
        '```tool\n{"tool": "report", "args": {"text": "seen"}}\n```',
        "Final answer: ok.",
    ])
    agent = Agent(backend, default_registry())
    agent.run("look", image_path=img)
    # first user turn carries the image (list content), later turns are text
    first_user = backend.calls[0][1]
    assert isinstance(first_user["content"], list)
    assert first_user["content"][0]["type"] == "image_url"
    second_round_last = backend.calls[1][-1]
    assert isinstance(second_round_last["content"], str)


def test_token_usage_accumulates_across_steps() -> None:
    backend = ScriptedBackend([
        '```tool\n{"tool": "report", "args": {"text": "a"}}\n```',
        "Final answer.",
    ])
    agent = Agent(backend, default_registry())
    result = agent.run("count tokens")
    # 2 backend calls * (5 prompt, 2 completion) each
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 4


def test_custom_tool_registry_dispatch() -> None:
    hits = []
    reg = ToolRegistry([Tool("ping", "ping tool", lambda a: hits.append(a) or "pong")])
    backend = ScriptedBackend([
        '```tool\n{"tool": "ping", "args": {"n": 1}}\n```',
        "Final.",
    ])
    agent = Agent(backend, reg)
    result = agent.run("ping it")
    assert hits == [{"n": 1}]
    assert result.steps[0].tool_result.output == "pong"
