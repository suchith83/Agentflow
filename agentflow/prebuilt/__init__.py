"""Prebuilt tools and agent packages for Agentflow.

Import concrete agent implementations from ``agentflow.prebuilt.agent`` and
tool helpers from ``agentflow.prebuilt.tools``.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any as _Any


_AGENT_EXPORTS = {
    "BaseReranker",
    "CohereReranker",
    "CrossEncoderReranker",
    "PlanActReflectAgent",
    "RAGAgent",
    "ReactAgent",
    "StructuredOutputAgent",
    "SupervisorTeamAgent",
    "SwarmAgent",
    "SwarmMemberConfig",
    "WorkerConfig",
}

_TOOL_EXPORTS = {
    "create_handoff_tool",
    "fetch_url",
    "file_read",
    "file_search",
    "file_write",
    "google_web_search",
    "is_handoff_tool",
    "make_agent_memory_tool",
    "make_user_memory_tool",
    "memory_tool",
    "safe_calculator",
    "vertex_ai_search",
}


def __getattr__(name: str) -> _Any:
    """Load prebuilt agents and tools only when explicitly requested."""
    if name == "agent":
        module = _import_module(f"{__name__}.agent")
        globals()[name] = module
        return module

    if name == "tools":
        module = _import_module(f"{__name__}.tools")
        globals()[name] = module
        return module

    if name in _AGENT_EXPORTS:
        agent_module = _import_module(f"{__name__}.agent")
        value = getattr(agent_module, name)
        globals()[name] = value
        return value

    if name in _TOOL_EXPORTS:
        tools_module = _import_module(f"{__name__}.tools")
        value = getattr(tools_module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PlanActReflectAgent",
    "RAGAgent",
    "ReactAgent",
    "RouterAgent",
    "StructuredOutputAgent",
    "SwarmAgent",
    "SwarmMemberConfig",
    # Agents
    "agent",
    # Tools
    "create_handoff_tool",
    "fetch_url",
    "file_read",
    "file_search",
    "file_write",
    "google_web_search",
    "is_handoff_tool",
    "make_agent_memory_tool",
    "make_user_memory_tool",
    "memory_tool",
    "safe_calculator",
    "tools",
    "vertex_ai_search",
]
