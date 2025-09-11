# Default message representation
import asyncio
import logging
from collections.abc import Awaitable
from datetime import datetime
import re
from typing import Any, Literal
from uuid import uuid4

from injectq import InjectQ
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


def generate_id(default_id: str | int | None) -> str | int:
    id_type = InjectQ.get_instance().try_get("generated_id_type", "string")
    generated_id = InjectQ.get_instance().try_get("generated_id", None)

    # if user provided an awaitable, resolve it
    if isinstance(generated_id, Awaitable):

        async def wait_for_id():
            return await generated_id

        generated_id = asyncio.run(wait_for_id())

    if generated_id:
        return generated_id

    if default_id:
        if id_type == "string" and isinstance(default_id, str):
            return default_id
        if id_type in ("int", "bigint") and isinstance(default_id, int):
            return default_id

    # if not matched or default_id is None, generate new id
    logger.debug(
        "Generating new id of type: %s. Default ID not provided or not matched %s",
        id_type,
        default_id,
    )

    if id_type == "int":
        return uuid4().int >> 32
    if id_type == "bigint":
        return uuid4().int >> 64
    return str(uuid4())


class TokenUsages(BaseModel):
    """
    Tracks token usage statistics for a message or model response.

    Example:
        >>> usage = TokenUsages(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        {'completion_tokens': 10, 'prompt_tokens': 20, 'total_tokens': 30, ...}
    """

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class Message(BaseModel):
    """
    Represents a message in a conversation, including content, role, metadata, and token usage.

    Example:
        >>> msg = Message(message_id="abc123", role="user", content="Hello!")
        {'message_id': 'abc123', 'role': 'user', 'content': 'Hello!', ...}

    Attributes:
        message_id (str): Unique identifier for the message.
        role (Literal["user", "assistant", "system", "tool"]): The role of the message sender.
        content (str): The message content.
        tools_calls (list[dict[str, Any]] | None): Tool call information, if any.
        tool_call_id (str | None): Tool call identifier, if any.
        function_call (dict[str, Any] | None): Function call information, if any.
        reasoning (str | None): Reasoning or explanation, if any.
        timestamp (datetime | None): Timestamp of the message.
        metadata (dict[str, Any]): Additional metadata.
        usages (TokenUsages | None): Token usage statistics.
        raw (dict[str, Any] | None): Raw data, if any.
    """

    message_id: str | int
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tools_calls: list[dict[str, Any]] | None = None
    # Mainly used for reply to tool calls
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None
    reasoning: str | None = None
    timestamp: datetime | None = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    usages: TokenUsages | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_text(
        cls,
        data: str,
        role: Literal["user", "assistant", "system", "tool"] = "user",
        message_id: str | None = None,
    ) -> "Message":
        """
        Create a Message instance from plain text.

        Args:
            data (str): The message content.
            role (Literal["user", "assistant", "system", "tool"]): The role of the sender.
            message_id (str | None): Optional message ID.

        Returns:
            Message: The created Message instance.
        """
        logger.debug("Creating message from text with role: %s", role)
        return cls(
            message_id=generate_id(message_id),
            role=role,
            content=data,
            timestamp=datetime.now(),
            metadata={},
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "Message":
        """
        Create a Message instance from a dictionary.

        Args:
            data (dict[str, Any]): Dictionary containing message data.

        Returns:
            Message: The created Message instance.

        Raises:
            ValueError: If required fields are missing.
        """
        # add Checks for required fields
        if "role" not in data or "content" not in data:
            logger.error("Missing required fields in data: %s", data)
            raise ValueError("Missing required fields: 'role' and 'content'")

        logger.debug("Creating message from dict with role: %s", data.get("role"))

        # Handle timestamp parsing
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        # Handle usages parsing
        usages = None
        if "usages" in data:
            usages = TokenUsages.model_validate(data["usages"])

        return cls(
            message_id=generate_id(data.get("message_id")),
            role=data.get("role", ""),
            content=data.get("content", ""),
            reasoning=data.get("reasoning"),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            usages=usages,
            raw=data.get("raw"),
        )

    @classmethod
    def from_response(cls, response: ModelResponse):
        """
        Create a Message instance from a ModelResponse object.

        Args:
            response (ModelResponse): The model response object.

        Returns:
            Message: The created Message instance.
        """
        data = response.model_dump()

        usages_data = data.get("usage", {})

        usages = TokenUsages(
            completion_tokens=usages_data.get("completion_tokens", 0),
            prompt_tokens=usages_data.get("prompt_tokens", 0),
            total_tokens=usages_data.get("total_tokens", 0),
            cache_creation_input_tokens=usages_data.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=usages_data.get("cache_read_input_tokens", 0),
            reasoning_tokens=usages_data.get("prompt_tokens_details", {}).get(
                "reasoning_tokens", 0
            ),
        )

        created_date = data.get("created", datetime.now())

        # check tools calls
        tools_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])

        tool_call_id = tools_calls[0].get("id") if tools_calls else None

        logger.debug("Creating message from model response with id: %s", response.id)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            content = ""
        return cls(
            message_id=generate_id(response.id),
            role="assistant",
            content=content,
            reasoning=data.get("choices", [{}])[0].get("message", {}).get("reasoning_content", ""),
            timestamp=created_date,
            metadata={
                "model": data.get("model", ""),
                "finish_reason": data.get("choices", [{}])[0].get("finish_reason", "UNKNOWN"),
                "object": data.get("object", ""),
                "prompt_tokens_details": usages_data.get("prompt_tokens_details", {}),
                "completion_tokens_details": usages_data.get("completion_tokens_details", {}),
            },
            usages=usages,
            raw=data,
            tools_calls=tools_calls if tools_calls else None,
            tool_call_id=tool_call_id,
        )

    @classmethod
    def tool_message(
        cls,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
        message_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ):
        """
        Create a tool message, optionally marking it as an error.

        Args:
            tool_call_id (str): The tool call identifier.
            content (str): The message content.
            is_error (bool): Whether this message represents an error.
            message_id (str | None): Optional message ID.
            meta (dict[str, Any] | None): Optional metadata.

        Returns:
            Message: The created tool message instance.
        """
        res = content
        if is_error:
            res = '{"success": False, "error": content}'

        logger.debug("Creating tool message with tool_call_id: %s", tool_call_id)
        msg_id = generate_id(message_id)
        return cls(
            message_id=msg_id,
            role="tool",
            content=res,
            timestamp=datetime.now(),
            metadata=meta or {},
            tool_call_id=tool_call_id,
        )

    @classmethod
    def create(
        cls,
        role: Literal["user", "assistant", "system", "tool"] = "assistant",
        content: str = "",
        reasoning: str = "",
        message_id: str | None = None,
        tool_call_id: str | None = None,
        tools_calls: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
        raw: dict[str, Any] | None = None,
    ):
        """
        Create a tool message, optionally marking it as an error.

        Args:
            tool_call_id (str): The tool call identifier.
            content (str): The message content.
            is_error (bool): Whether this message represents an error.
            message_id (str | None): Optional message ID.
            meta (dict[str, Any] | None): Optional metadata.

        Returns:
            Message: The created tool message instance.
        """
        msg_id = generate_id(message_id)
        return cls(
            message_id=msg_id,
            role=role,
            content=content,
            reasoning=reasoning,
            timestamp=datetime.now(),
            metadata=meta or {},
            tools_calls=tools_calls,
            tool_call_id=tool_call_id,
            raw=raw,
        )
