"""Tests for tool-call parsing + registry dispatch."""

from __future__ import annotations

import pytest

from swift_agent.example_tools import default_registry
from swift_agent.tools import Tool, ToolCall, ToolRegistry, parse_tool_call


# ---------------------------------------------------------------------------
# parse_tool_call
# ---------------------------------------------------------------------------


def test_parse_fenced_tool_block() -> None:
    text = (
        "I see a mug on the left.\n"
        '```tool\n{"tool": "move", "args": {"direction": "left", "distance": 0.5}}\n```'
    )
    call = parse_tool_call(text)
    assert call == ToolCall(tool="move", args={"direction": "left", "distance": 0.5})


def test_parse_returns_none_for_final_answer() -> None:
    """A reply with no tool block is a final answer -> None."""
    assert parse_tool_call("The image shows a red mug. Final answer: a mug.") is None


def test_parse_first_block_only_when_multiple() -> None:
    text = (
        '```tool\n{"tool": "report", "args": {"text": "first"}}\n```\n'
        '```tool\n{"tool": "move", "args": {"direction": "up"}}\n```'
    )
    call = parse_tool_call(text)
    assert call.tool == "report"
    assert call.args == {"text": "first"}


def test_parse_bare_object_without_fence() -> None:
    """Some models drop the fence; a trailing bare {"tool":...} still parses."""
    text = 'Let me grasp it.\n{"tool": "grasp", "args": {"target": "mug"}}'
    call = parse_tool_call(text)
    assert call.tool == "grasp"
    assert call.args == {"target": "mug"}


def test_parse_malformed_json_raises() -> None:
    text = '```tool\n{"tool": "move", "args": {bad json}}\n```'
    with pytest.raises(ValueError, match="Malformed tool-call JSON"):
        parse_tool_call(text)


def test_parse_missing_tool_key_raises() -> None:
    text = '```tool\n{"args": {"direction": "left"}}\n```'
    with pytest.raises(ValueError, match="missing 'tool' key"):
        parse_tool_call(text)


def test_parse_non_object_args_raises() -> None:
    text = '```tool\n{"tool": "move", "args": [1, 2, 3]}\n```'
    with pytest.raises(ValueError, match="must be an object"):
        parse_tool_call(text)


def test_parse_defaults_args_to_empty_dict() -> None:
    call = parse_tool_call('```tool\n{"tool": "report"}\n```')
    assert call == ToolCall(tool="report", args={})


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_registry_dispatch_success() -> None:
    reg = default_registry()
    result = reg.dispatch(ToolCall("move", {"direction": "left", "distance": 0.5}))
    assert result.ok
    assert result.output == {"command": "move", "direction": "left", "distance_m": 0.5}


def test_registry_unknown_tool_is_failed_result_not_raise() -> None:
    reg = default_registry()
    result = reg.dispatch(ToolCall("teleport", {}))
    assert not result.ok
    assert "unknown tool" in result.error
    assert "teleport" in result.error


def test_registry_tool_exception_becomes_failed_result() -> None:
    """A tool raising (bad args) is caught and surfaced, not propagated."""
    reg = default_registry()
    result = reg.dispatch(ToolCall("move", {"direction": "sideways"}))
    assert not result.ok
    assert "direction must be one of" in result.error


def test_registry_rejects_duplicate_registration() -> None:
    reg = ToolRegistry()
    reg.register(Tool("x", "desc", lambda a: a))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Tool("x", "desc2", lambda a: a))


def test_registry_describe_lists_tools() -> None:
    desc = default_registry().describe()
    assert "move:" in desc and "grasp:" in desc and "report:" in desc


def test_registry_describe_empty() -> None:
    assert ToolRegistry().describe() == "(no tools available)"


def test_tool_result_as_observation_formats() -> None:
    reg = default_registry()
    ok = reg.dispatch(ToolCall("report", {"text": "hi"}))
    assert ok.as_observation().startswith("[tool:report] OK ->")
    bad = reg.dispatch(ToolCall("nope", {}))
    assert bad.as_observation().startswith("[tool:nope] ERROR ->")
