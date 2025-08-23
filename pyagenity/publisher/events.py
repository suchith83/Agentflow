from enum import StrEnum
from typing import Any


# from pydantic import BaseModel


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
    ERROR = "error"
    CUSTOM = "custom"


class Event:
    """Represents an event."""

    source: SourceType
    event_type: EventType
    config: dict[str, Any]
    payload: dict[str, Any]
    meta: dict[str, Any]
