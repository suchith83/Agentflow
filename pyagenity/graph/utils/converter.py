from typing import Any

from pyagenity.graph.state import AgentState

from .message import Message


def _convert_dict(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "role": message.role,
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }

    if message.role == "assistant" and message.tools_calls:
        return {
            "role": message.role,
            "content": message.content if message.content else "",
            "tool_calls": message.tools_calls,
        }

    return {"role": message.role, "content": message.content}


def convert_messages(
    system_prompts: list[dict[str, Any]],
    state: AgentState | None = None,
    extra_messages: list[Message] | None = None,
) -> list[dict[str, Any]]:
    if system_prompts is None:
        raise ValueError("System prompts cannot be None")

    res = []
    res += system_prompts

    if state and state.context_summary:
        summary = {
            "role": "assistant",
            "content": state.context_summary if state.context_summary else "",
        }
        res.append(summary)

    if state and state.context:
        for msg in state.context:
            res.append(_convert_dict(msg))

    # now add current messages
    if extra_messages:
        for msg in extra_messages:
            res.append(_convert_dict(msg))

    return res
