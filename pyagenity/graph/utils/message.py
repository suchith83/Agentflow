# Default message representation
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Literal


@dataclass
class TokenUsages:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TokenUsages":
        """Create from dictionary."""
        return TokenUsages(
            completion_tokens=data.get("completion_tokens", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cache_creation_input_tokens=data.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=data.get("cache_read_input_tokens", 0),
        )


@dataclass
class Message:
    """A message in the conversation."""

    message_id: str
    role: Literal[
        "human", "assistant", "system", "tool"
    ]  # "human", "assistant", "system"
    content: str
    # reasoning behind the message
    reasoning: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = None
    usages: Optional[TokenUsages] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with all fields."""
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
            "usages": self.usages.to_dict() if self.usages else None,
        }

    @staticmethod
    def from_dict(
        data: dict[str, Any],
    ) -> "Message":
        """Create from dictionary."""
        # add Checks for required fields
        if "role" not in data or "content" not in data:
            raise ValueError("Missing required fields: 'role' and 'content'")

        return Message(
            message_id=data.get("message_id", ""),
            role=data.get("role", ""),
            content=data.get("content", ""),
            reasoning=data.get("reasoning"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(),
            metadata=data.get("metadata", {}),
            usages=TokenUsages.from_dict(data["usages"]) if "usages" in data else None,
        )
