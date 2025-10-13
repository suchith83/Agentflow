"""
Message conversion utilities for TAF agent graphs.

This module provides helpers to convert Message objects and agent state
into dicts suitable for LLM and tool invocation payloads.
"""

import logging
from typing import TYPE_CHECKING, Any, Union

from agentflow.state.message import Message
from agentflow.state.message_block import RemoteToolCallBlock, ToolResultBlock


if TYPE_CHECKING:
    from agentflow.state import AgentState

logger = logging.getLogger(__name__)


def _convert_dict(message: Message) -> dict[str, Any] | None:
    """
    Convert a Message object to a dictionary for LLM/tool payloads.

    Args:
        message (Message): The message to convert.

    Returns:
        dict[str, Any]: Dictionary representation of the message.
    """
    # if any remote tool call exists we are skipping the tool result block
    # as remote tool calls are not supported in the current implementation
    if RemoteToolCallBlock in message.content:
        return None

    if message.role == "tool":
        content = message.content
        call_id = ""
        for i in content:
            if isinstance(i, ToolResultBlock):
                call_id = i.call_id
                break

        return {
            "role": message.role,
            "content": message.text(),
            "tool_call_id": call_id,
        }

    if message.role == "assistant" and message.tools_calls:
        return {
            "role": message.role,
            "content": message.text(),
            "tool_calls": message.tools_calls,
        }

    return {"role": message.role, "content": message.text()}


def convert_messages(
    system_prompts: list[dict[str, Any]],
    state: Union["AgentState", None] = None,
    extra_messages: list[Message] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert system prompts, agent state, and extra messages to a list of dicts for
    LLM/tool payloads.

    Args:
        system_prompts (list[dict[str, Any]]): List of system prompt dicts.
        state (AgentState | None): Optional agent state containing context and summary.
        extra_messages (list[Message] | None): Optional extra messages to include.

    Returns:
        list[dict[str, Any]]: List of message dicts for payloads.

    Raises:
        ValueError: If system_prompts is None.
    """
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
            formatted = _convert_dict(msg)
            if formatted:
                res.append(formatted)

    if extra_messages:
        for msg in extra_messages:
            formatted = _convert_dict(msg)
            if formatted:
                res.append(formatted)

    logger.debug("Number of Converted messages: %s", len(res))
    return res
