import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    """Sources of events."""

    MESSAGE = "message"
    GRAPH = "graph"
    NODE = "node"
    STATE = "state"
    TOOL = "tool"


class EventType(StrEnum):
    """Types of events."""

    CHANGED = "changed"
    INITIALIZE = "initialize"
    INVOKED = "invoked"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    RUNNING = "running"
    ERROR = "error"
    CUSTOM = "custom"


class Event(BaseModel):
    """Represents an event."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    source: SourceType
    event_type: EventType
    config: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        return f"Event(id={self.id}, source={self.source}, event_type={self.event_type})"
