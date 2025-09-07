import enum
import time  # For timestamp generation
import uuid  # For reference_id generation
from typing import Any, Literal

from pydantic import BaseModel, Field


class StreamEvent(enum.StrEnum):
    """
    Enum representing different types of stream events.

    Example:
        >>> event = StreamEvent.DATA
        StreamEvent.DATA
    """

    NODE_EXECUTION = "node_execution"  # Before Node Execution
    TOKEN = "token"  # noqa: S105  # Streaming Tokens
    MESSAGE = "message"  # Final Message
    TOOL_EXECUTION = "tool_execution"  # Before Tool Execution
    TOOL_RESULT = "tool_result"  # Tool Result
    MCP_TOOL_EXECUTION = "mcp_tool_execution"  # Before MCP Tool Execution
    MCP_TOOL_RESULT = "mcp_tool_result"  # MCP Tool Result
    INTERRUPTED = "interrupted"  # Before Interrupted
    NODE = "node"  # Which node is being executed or when changed
    STATE = "state"  # Current State
    CONTEXT_TRIMMING = "context_trimming"  # When context is being trimmed
    ERROR = "error"  # If any error occurs
    COMPLETE = "complete"  # When everything is done


class StreamChunks(BaseModel):
    """Represents chunks of streamed data with event information.

    Defaults are provided via Pydantic Field with default_factory to avoid
    mutable default arguments and to generate runtime values for timestamp
    and reference_id.

    Example:
        >>> chunk = StreamChunks()
        >>> chunk.event
        <StreamEvent.message: 'message'>
        >>> chunk.event_type
        'Before'
    """

    event: StreamEvent = Field(default=StreamEvent.MESSAGE)
    event_type: Literal["Before", "After"] = Field(default="Before")
    data: dict[str, Any] = Field(default_factory=dict)
    reference_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)
