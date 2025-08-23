import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    """
    Enum representing the sources of events.

    Example:
        >>> from pyagenity.publisher.events import SourceType
        >>> SourceType.MESSAGE
        <SourceType.MESSAGE: 'message'>
    """

    MESSAGE = "message"
    GRAPH = "graph"
    NODE = "node"
    STATE = "state"
    TOOL = "tool"


class EventType(StrEnum):
    """
    Enum representing the types of events.

    Example:
        >>> from pyagenity.publisher.events import EventType
        >>> EventType.ERROR
        <EventType.ERROR: 'error'>
    """

    CHANGED = "changed"
    INITIALIZE = "initialize"
    INVOKED = "invoked"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    RUNNING = "running"
    ERROR = "error"
    CUSTOM = "custom"


class Event(BaseModel):
    """
    Represents an event in the system.

    This class encapsulates all information about an event, including its source, type,
    configuration, payload, and metadata. Events are uniquely identified by an ID and timestamp.

    Example:
        >>> from pyagenity.publisher.events import Event, SourceType, EventType
        >>> event = Event(source=SourceType.MESSAGE, event_type=EventType.INVOKED)
        >>> print(event)
        Event(id=..., source=SourceType.MESSAGE, event_type=EventType.INVOKED)

    Attributes:
        id (str): Unique identifier for the event.
        timestamp (str): ISO-formatted timestamp of event creation.
        source (SourceType): The source of the event.
        event_type (EventType): The type of the event.
        config (dict[str, Any]): Configuration related to the event.
        payload (dict[str, Any]): Payload data for the event.
        meta (dict[str, Any]): Additional metadata for the event.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    source: SourceType
    event_type: EventType
    config: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """
        Create an Event instance from a dictionary.

        Args:
            data (dict[str, Any]): Dictionary containing event data.

        Returns:
            Event: An Event instance populated with the provided data.
        """
        return cls(**data)

    def __repr__(self) -> str:
        """
        Return a string representation of the Event.

        Returns:
            str: String representation of the Event instance.
        """
        return f"Event(id={self.id}, source={self.source}, event_type={self.event_type})"
