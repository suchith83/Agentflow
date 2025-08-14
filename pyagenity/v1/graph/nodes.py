from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..agent.agent import Agent, AgentResponse


class NodeExecutionError(Exception):
    pass


@dataclass
class BaseNode:
    name: str
    description: str | None = None

    def run(self, state: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class LLMNode(BaseNode):
    agent: Agent | None = field(default=None)
    prompt_builder: Callable[[dict[str, Any]], str] | None = None
    output_key: str = "last_response"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.agent is None:
            raise NodeExecutionError("LLMNode requires an Agent instance")
        if self.prompt_builder:
            prompt = self.prompt_builder(state)
        else:
            # Fallback to last user input or an empty string
            prompt = state.get("last_user_input", "")
        result = self.agent.run(prompt)
        if isinstance(result, AgentResponse):
            # Append assistant message to a conversation log
            messages = state.setdefault("messages", [])
            messages.append({"role": "assistant", "content": result.content})
            state[self.output_key] = result.content
            state[f"{self.output_key}_usage"] = result.usage
        else:
            state[self.output_key] = str(result)
        return state


@dataclass
class FunctionNode(BaseNode):
    func: Callable[[dict[str, Any]], dict[str, Any]] | None = field(default=None)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.func is None:
            raise NodeExecutionError("FunctionNode requires a callable 'func'")
        updated = self.func(state)
        if updated is not None:
            state.update(updated)
        return state


@dataclass
class HumanInputNode(BaseNode):
    input_key: str = "human_input"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        # Executor will detect absence of human input and pause
        return state
