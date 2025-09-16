import enum
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class Event(str, enum.Enum):
    NODE_EXECUTION = "node_execution"
    TOOL_EXECUTION = "tool_execution"
    MESSAGE = "message"
    STATE = "state"
    ERROR = "error"
    COMPLETE = "complete"


class EventType(str, enum.Enum):
    START = "start"
    PROGRESS = "progress"
    END = "end"
    UPDATE = "update"


class ContentType(str, enum.Enum):
    TEXT = "text"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE = "state"
    ERROR = "error"


class EventModel(BaseModel):
    """
    Represents a chunk of streamed data with event and content semantics.

    Designed for consistent and structured real-time streaming of execution updates, tool calls,
    state changes, messages, and errors.

    Supports both delta (incremental) and full content.
    """

    # Event metadata
    event: Event = Field(..., description="Type of the event source")
    event_type: EventType = Field(
        ..., description="Phase of the event (start, progress, end, update)"
    )

    # Streamed content
    content: str = Field(default="", description="Streamed textual content")
    delta: bool = Field(default=False, description="True if this is a delta update (incremental)")

    # Data payload
    data: dict[str, Any] = Field(default_factory=dict, description="Additional structured data")

    # Metadata
    content_type: ContentType | None = Field(default=None, description="Semantic type of content")
    sequence_id: int = Field(default=0, description="Monotonic sequence ID for stream ordering")
    node_name: str = Field(default="", description="Name of the node producing this chunk")
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this stream/run"
    )
    timestamp: float = Field(
        default_factory=time.time, description="UNIX timestamp of when chunk was created"
    )
    is_error: bool = Field(
        default=False, description="Marks this chunk as representing an error state"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional metadata for consumers"
    )

    class Config:
        use_enum_values = True  # Output enums as strings
