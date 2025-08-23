# Default message representation
import logging
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from litellm.types.utils import ModelResponse
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class TokenUsages(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenUsages":
        """Create from dictionary."""
        return cls(**data)


class Message(BaseModel):
    """A message in the conversation."""

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

    def to_dict(self, include_raw: bool = False) -> dict[str, Any]:
        """Convert to dictionary with all fields. Handles both datetime and int timestamps."""
        data = self.model_dump()

        # Handle timestamp formatting
        ts = data.get("timestamp")
        if ts is None:
            ts_val = None
        elif hasattr(ts, "isoformat"):
            ts_val = ts.isoformat()
        elif isinstance(ts, int):
            ts_val = ts  # leave as int (epoch)
        else:
            ts_val = str(ts)

        result = {
            "message_id": data["message_id"],
            "role": data["role"],
            "content": data["content"],
            "reasoning": data["reasoning"],
            "timestamp": ts_val,
            "metadata": data["metadata"],
            "usages": self.usages.to_dict() if self.usages else None,
        }

        if include_raw:
            result["raw"] = data["raw"]

        return result

    @classmethod
    def from_text(
        cls,
        data: str,
        role: Literal["user", "assistant", "system", "tool"] = "user",
        message_id: str | None = None,
    ) -> "Message":
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
        """Create from dictionary."""
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
            usages = TokenUsages.from_dict(data["usages"])

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

        # 1734366691
        created_date = data.get("created", datetime.now())

        # check tools calls
        tools_calls = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])

        tool_call_id = tools_calls[0].get("id") if tools_calls else None

        logger.debug("Creating message from model response with id: %s", response.id)
        # TODO: replace this with int id
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
