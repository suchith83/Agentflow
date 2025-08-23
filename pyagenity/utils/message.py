# Default message representation
import logging
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from litellm.types.utils import ModelResponse
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


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

    message_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tools_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None
    reasoning: str | None = None
    timestamp: datetime | None = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    usages: TokenUsages | None = None
    raw: dict[str, Any] | None = None

    # def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
    #     """
    #     Convert the Message instance to a dictionary with all fields.

    #     Args:
    #         include_raw (bool): Whether to include the raw field in the output.

    #     Returns:
    #         dict[str, Any]: Dictionary representation of the message.
    #     """
    #     data = self.model_dump()

    #     # Handle timestamp formatting
    #     ts = data.get("timestamp")
    #     if ts is None:
    #         ts_val = None
    #     elif hasattr(ts, "isoformat"):
    #         ts_val = ts.isoformat()
    #     elif isinstance(ts, int):
    #         ts_val = ts  # leave as int (epoch)
    #     else:
    #         ts_val = str(ts)

    #     result = {
    #         "message_id": data["message_id"],
    #         "role": data["role"],
    #         "content": data["content"],
    #         "reasoning": data["reasoning"],
    #         "timestamp": ts_val,
    #         "metadata": data["metadata"],
    #         "usages": self.usages.to_dict() if self.usages else None,
    #     }

    #     if include_raw:
    #         result["raw"] = data["raw"]

    #     return result

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
            message_id=message_id or str(uuid4()),  # Generate a new UUID
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
            message_id=data.get("message_id", str(uuid4())),
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
        return cls(
            message_id=response.id,
            role="assistant",
            content=data.get("choices", [{}])[0]
            .get("message", {})
            .get(
                "content",
                "",
            ),
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
        return cls(
            message_id=message_id or str(uuid4()),
            role="tool",
            content=res,
            timestamp=datetime.now(),
            metadata=meta or {},
            tool_call_id=tool_call_id,
        )
