import enum
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from pyagenity.utils.message import ContentBlock


class Event(str, enum.Enum):
    # All the event source
    GRAPH_EXECUTION = "graph_execution"
    NODE_EXECUTION = "node_execution"
    TOOL_EXECUTION = "tool_execution"
    STREAMING = "streaming"


class EventType(str, enum.Enum):
    START = "start"
    PROGRESS = "progress"
    RESULT = "result"
    END = "end"
    UPDATE = "update"
    ERROR = "error"
    INTERRUPTED = "interrupted"


class ContentType(str, enum.Enum):
    TEXT = "text"
    MESSAGE = "message"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    DATA = "data"
    STATE = "state"
    UPDATE = "update"
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
    # Structured content blocks for multimodal/structured streaming
    content_blocks: list[ContentBlock] | None = Field(
        default=None, description="Structured content blocks carried by this event"
    )
    # Delta controls
    delta: bool = Field(default=False, description="True if this is a delta update (incremental)")
    delta_type: Literal["text", "json", "binary"] | None = Field(
        default=None, description="Type of delta when delta=True"
    )
    block_index: int | None = Field(
        default=None, description="Index of the content block this chunk applies to"
    )
    chunk_index: int | None = Field(default=None, description="Per-block chunk index for ordering")
    byte_offset: int | None = Field(
        default=None, description="Byte offset for binary/media streaming"
    )

    # Data payload
    data: dict[str, Any] = Field(default_factory=dict, description="Additional structured data")

    # Metadata
    content_type: list[ContentType] | None = Field(
        default=None, description="Semantic type of content"
    )
    sequence_id: int = Field(default=0, description="Monotonic sequence ID for stream ordering")
    node_name: str = Field(default="", description="Name of the node producing this chunk")
    run_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this stream/run"
    )
    thread_id: str | int = Field(default="", description="Thread ID for this execution")
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

    @classmethod
    def default(
        cls,
        base_config: dict,
        data: dict[str, Any],
        content_type: list[ContentType],
        event: Event = Event.GRAPH_EXECUTION,
        event_type=EventType.START,
        node_name: str = "",
        extra: dict[str, Any] | None = None,
    ) -> "EventModel":
        """Create a default EventModel instance with minimal required fields."""
        thread_id = base_config.get("thread_id", "")
        run_id = base_config.get("run_id", "")

        metadata = {
            "run_timestamp": base_config.get("timestamp", ""),
            "user_id": base_config.get("user_id"),
            "is_stream": base_config.get("is_stream", False),
        }
        if extra:
            metadata.update(extra)
        return cls(
            event=event,
            event_type=event_type,
            delta=False,
            content_type=content_type,
            data=data,
            thread_id=thread_id,
            node_name=node_name,
            run_id=run_id,
            metadata=metadata,
        )

    @classmethod
    def stream(
        cls,
        base_config: dict,
        node_name: str = "",
        extra: dict[str, Any] | None = None,
    ) -> "EventModel":
        """Create a default EventModel instance with minimal required fields."""
        thread_id = base_config.get("thread_id", "")
        run_id = base_config.get("run_id", "")

        metadata = {
            "run_timestamp": base_config.get("timestamp", ""),
            "user_id": base_config.get("user_id"),
            "is_stream": base_config.get("is_stream", False),
        }
        if extra:
            metadata.update(extra)
        return cls(
            event=Event.STREAMING,
            event_type=EventType.UPDATE,
            delta=True,
            content_type=[ContentType.TEXT, ContentType.REASONING],
            data={},
            thread_id=thread_id,
            node_name=node_name,
            run_id=run_id,
            metadata=metadata,
        )
