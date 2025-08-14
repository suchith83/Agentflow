# Default message representation
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Literal


@dataclass
class Message:
    """A message in the conversation."""

    role: Literal[
        "human", "assistant", "system", "tool"
    ]  # "human", "assistant", "system"
    content: str
    timestamp: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

    def from_dict(
        self,
        data: dict[str, Any],
    ) -> "Message":
        """Create from dictionary."""
        # add Checks for required fields
        if "role" not in data or "content" not in data:
            raise ValueError("Missing required fields: 'role' and 'content'")

        return Message(
            role=data.get("role", ""),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(),
            metadata=data.get("metadata", {}),
        )
