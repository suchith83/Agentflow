from __future__ import annotations
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from ..agent.agent import Agent, AgentResponse


class NodeExecutionError(Exception):
    pass


@dataclass
class BaseNode:
    name: str
    description: Optional[str] = None

    def run(
        self, state: Dict[str, Any]
    ) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class LLMNode(BaseNode):
    agent: Optional[Agent] = field(default=None)
    prompt_builder: Optional[Callable[[Dict[str, Any]], str]] = None
    output_key: str = "last_response"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
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
    func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = field(default=None)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if self.func is None:
            raise NodeExecutionError("FunctionNode requires a callable 'func'")
        updated = self.func(state)
        if updated is not None:
            state.update(updated)
        return state


@dataclass
class HumanInputNode(BaseNode):
    input_key: str = "human_input"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Executor will detect absence of human input and pause
        return state
