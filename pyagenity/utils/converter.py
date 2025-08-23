import logging
from typing import TYPE_CHECKING, Any, Union

from .message import Message


if TYPE_CHECKING:
    from pyagenity.state import AgentState


logger = logging.getLogger(__name__)


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
    state: Union["AgentState", None] = None,
    extra_messages: list[Message] | None = None,
) -> list[dict[str, Any]]:
    if system_prompts is None:
        logger.error("System prompts are None")
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

    logger.debug("Number of Converted messages: %s", len(res))
    return res
