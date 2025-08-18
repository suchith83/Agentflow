"""Streaming utilities for handling both streamable and non-streamable responses."""

import asyncio
import time
from collections.abc import AsyncIterator, Generator
from uuid import uuid4

from litellm.types.utils import ModelResponse

from .message import Message


class StreamChunk:
    """Represents a chunk of streaming content."""

    def __init__(
        self,
        content: str = "",
        delta: str = "",
        finish_reason: str | None = None,
        tool_calls: list | None = None,
        role: str = "assistant",
        is_final: bool = False,
    ):
        self.content = content  # Full content so far
        self.delta = delta  # New content in this chunk
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls
        self.role = role
        self.is_final = is_final

    def to_message(self) -> Message:
        """Convert chunk to a Message object."""
        return Message(
            role=self.role,  # type: ignore
            content=self.content,
            tools_calls=self.tool_calls,
            message_id=str(uuid4()),
        )


def is_streaming_response(response) -> bool:
    """Check if a response is a streaming iterator from litellm."""
    # Check if it's an iterator and has streaming-like attributes
    return (
        hasattr(response, "__iter__")
        and hasattr(response, "__next__")
        and not isinstance(response, str | bytes | dict | list)
        and not isinstance(response, ModelResponse)
    )


def is_async_streaming_response(response) -> bool:
    """Check if a response is an async streaming iterator from litellm."""
    return (
        hasattr(response, "__aiter__")
        and hasattr(response, "__anext__")
        and not isinstance(response, str | bytes | dict | list)
        and not isinstance(response, ModelResponse)
    )


def chunk_text(text: str, chunk_size: int = 5) -> list[str]:
    """Split text into word-based chunks for simulated streaming."""
    if not text:
        return [""]

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        if i + chunk_size < len(words):
            chunk += " "  # Add space if not the last chunk
        chunks.append(chunk)

    return chunks if chunks else [""]


def simulate_streaming(content: str, delay: float = 0.05) -> Generator[StreamChunk, None, None]:
    """Simulate streaming by chunking complete content."""
    if not content:
        yield StreamChunk(content="", delta="", is_final=True)
        return

    chunks = chunk_text(content)
    accumulated_content = ""

    for i, chunk in enumerate(chunks):
        accumulated_content += chunk
        is_final = i == len(chunks) - 1

        yield StreamChunk(
            content=accumulated_content,
            delta=chunk,
            is_final=is_final,
            finish_reason="stop" if is_final else None,
        )

        if not is_final and delay > 0:
            time.sleep(delay)


async def simulate_async_streaming(content: str, delay: float = 0.05) -> AsyncIterator[StreamChunk]:
    """Simulate async streaming by chunking complete content."""
    if not content:
        yield StreamChunk(content="", delta="", is_final=True)
        return

    chunks = chunk_text(content)
    accumulated_content = ""

    for i, chunk in enumerate(chunks):
        accumulated_content += chunk
        is_final = i == len(chunks) - 1

        yield StreamChunk(
            content=accumulated_content,
            delta=chunk,
            is_final=is_final,
            finish_reason="stop" if is_final else None,
        )

        if not is_final and delay > 0:
            await asyncio.sleep(delay)


def stream_from_litellm_response(response) -> Generator[StreamChunk, None, None]:
    """Convert a litellm streaming response to our StreamChunk format."""
    accumulated_content = ""

    for chunk in response:
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            delta_content = ""

            if hasattr(choice, "delta") and choice.delta:
                delta_content = getattr(choice.delta, "content", "") or ""

            accumulated_content += delta_content
            finish_reason = getattr(choice, "finish_reason", None)

            yield StreamChunk(
                content=accumulated_content,
                delta=delta_content,
                finish_reason=finish_reason,
                is_final=finish_reason is not None,
            )


async def astream_from_litellm_response(response) -> AsyncIterator[StreamChunk]:
    """Convert a litellm async streaming response to our StreamChunk format."""
    accumulated_content = ""

    async for chunk in response:
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            choice = chunk.choices[0]
            delta_content = ""

            if hasattr(choice, "delta") and choice.delta:
                delta_content = getattr(choice.delta, "content", "") or ""

            accumulated_content += delta_content
            finish_reason = getattr(choice, "finish_reason", None)

            yield StreamChunk(
                content=accumulated_content,
                delta=delta_content,
                finish_reason=finish_reason,
                is_final=finish_reason is not None,
            )


def extract_content_from_response(response) -> str:
    """Extract text content from various response types."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        # Try common content keys
        return response.get("content", "")
    if (
        isinstance(response, ModelResponse)
        and hasattr(response, "choices")
        and len(response.choices) > 0
    ):
        choice = response.choices[0]
        # Safely try to get message content
        if hasattr(choice, "message"):
            message = getattr(choice, "message", None)
            if message and hasattr(message, "content"):
                return getattr(message, "content", "") or ""
    if hasattr(response, "content"):
        return str(getattr(response, "content", ""))

    return str(response)
