"""
Stream chunk primitives for unified streaming data handling.

This module provides a unified StreamChunk class that can encapsulate different types
of streaming data (Messages, EventModels, etc.) in a type-safe manner.
This enables clean separation between conversation content and execution state while
providing a consistent interface for streaming consumers.

Classes:
    StreamChunk: Unified wrapper for streaming data with type discrimination.
"""

from typing import Literal

from pydantic import BaseModel

from .message import Message


class StreamChunk(BaseModel):
    """
    Unified wrapper for different types of streaming data.

    This class provides a single interface for handling various streaming chunk types
    (messages, events, state updates, errors) with type-safe discrimination.

    Attributes:
        type: The type of streaming chunk.
        data: The actual chunk data (Message, EventModel, dict, etc.).
        metadata: Optional additional metadata for the chunk.
    """

    type: Literal["message", "event", "state", "error"]
    message: Message | None = None
