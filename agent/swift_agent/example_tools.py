"""Example tools for the observe–reason–act agent.

These illustrate the structured tool protocol for the robotic /
small-batch inference scenarios described in the report (§3.3). They are
deliberately side-effect-free (they return structured dicts rather than
driving real hardware) so the demo + tests run anywhere. Swap ``fn`` for
real motor / API calls in a deployment.
"""

from __future__ import annotations

from typing import Any

from swift_agent.tools import Tool, ToolRegistry

_VALID_DIRECTIONS = {"left", "right", "forward", "backward", "up", "down"}


def _move(args: dict[str, Any]) -> dict[str, Any]:
    """Emit a structured motor command (observe→reason→**act**)."""
    direction = args.get("direction")
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {direction!r}"
        )
    distance = float(args.get("distance", 0.1))
    if distance <= 0:
        raise ValueError(f"distance must be positive, got {distance}")
    return {"command": "move", "direction": direction, "distance_m": distance}


def _grasp(args: dict[str, Any]) -> dict[str, Any]:
    """Emit a structured grasp command for a named/located object."""
    target = args.get("target")
    if not target:
        raise ValueError("grasp requires a 'target' (object name or description)")
    return {"command": "grasp", "target": str(target)}


def _report(args: dict[str, Any]) -> dict[str, Any]:
    """Record a structured observation report (a no-op 'note to self')."""
    text = args.get("text", "")
    return {"command": "report", "text": str(text)}


def default_registry() -> ToolRegistry:
    """A small registry covering the report's robotic-agent example."""
    return ToolRegistry([
        Tool(
            name="move",
            description=(
                'Move the robot. args: {"direction": '
                '"left|right|forward|backward|up|down", "distance": <meters>}'
            ),
            fn=_move,
        ),
        Tool(
            name="grasp",
            description='Grasp an object. args: {"target": "<object name or description>"}',
            fn=_grasp,
        ),
        Tool(
            name="report",
            description='Record an observation. args: {"text": "<what you observed>"}',
            fn=_report,
        ),
    ])


__all__ = ["default_registry"]
