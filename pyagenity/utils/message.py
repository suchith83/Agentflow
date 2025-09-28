"""
Message and content block primitives for agent graphs.

This module defines the core message representation, multimodal content blocks,
token usage tracking, and utility functions for agent graph communication.

Classes:
    TokenUsages: Tracks token usage statistics for a message or model response.
    MediaRef: Reference to media content (image/audio/video/document/data).
    AnnotationRef: Reference to annotation metadata.
    TextBlock, ImageBlock, AudioBlock, VideoBlock, DocumentBlock, DataBlock,
        ToolCallBlock, ToolResultBlock, ReasoningBlock, AnnotationBlock, ErrorBlock:
        Multimodal content primitives for message composition.
    Message: Represents a message in a conversation, including content, role, metadata, and token usage.

Functions:
    generate_id: Generates a message or tool call ID based on DI context and type.
"""

import asyncio
import logging
from collections.abc import Awaitable
from datetime import datetime
from typing import Annotated, Any, Literal, Union
from uuid import uuid4

from injectq import InjectQ
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


def generate_id(default_id: str | int | None) -> str | int:
    """
    Generate a message or tool call ID based on DI context and type.

    Args:
        default_id (str | int | None): Default ID to use if provided and matches type.

    Returns:
        str | int: Generated or provided ID, type determined by DI context.

    Raises:
        None

    Example:
        >>> generate_id("abc123")
        'abc123'
        >>> generate_id(None)
        'a-uuid-string'
    """
    id_type = InjectQ.get_instance().try_get("generated_id_type", "string")
    generated_id = InjectQ.get_instance().try_get("generated_id", None)

    # if user provided an awaitable, resolve it
    if isinstance(generated_id, Awaitable):

        async def wait_for_id():
            return await generated_id

        generated_id = asyncio.run(wait_for_id())

    if generated_id:
        return generated_id

    if default_id:
        if id_type == "string" and isinstance(default_id, str):
            return default_id
        if id_type in ("int", "bigint") and isinstance(default_id, int):
            return default_id

    # if not matched or default_id is None, generate new id
    logger.debug(
        "Generating new id of type: %s. Default ID not provided or not matched %s",
        id_type,
        default_id,
    )

    if id_type == "int":
        return uuid4().int >> 32
    if id_type == "bigint":
        return uuid4().int >> 64
    return str(uuid4())


class TokenUsages(BaseModel):
    """
    Tracks token usage statistics for a message or model response.

    Attributes:
        completion_tokens (int): Number of completion tokens used.
        prompt_tokens (int): Number of prompt tokens used.
        total_tokens (int): Total tokens used.
        reasoning_tokens (int): Reasoning tokens used (optional).
        cache_creation_input_tokens (int): Cache creation input tokens (optional).
        cache_read_input_tokens (int): Cache read input tokens (optional).
        image_tokens (int | None): Image tokens for multimodal models (optional).
        audio_tokens (int | None): Audio tokens for multimodal models (optional).

    Example:
        >>> usage = TokenUsages(completion_tokens=10, prompt_tokens=20, total_tokens=30)
        {'completion_tokens': 10, 'prompt_tokens': 20, 'total_tokens': 30, ...}
    """

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    reasoning_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # Optional modality-specific usage fields for multimodal models
    image_tokens: int | None = 0
    audio_tokens: int | None = 0


# --- Multimodal content primitives ---


class MediaRef(BaseModel):
    """
    Reference to media content (image/audio/video/document/data).

    Prefer referencing by URL or provider file_id over inlining base64 for large payloads.

    Attributes:
        kind (Literal["url", "file_id", "data"]): Type of reference.
        url (str | None): URL to media content.
        file_id (str | None): Provider-managed file ID.
        data_base64 (str | None): Base64-encoded data (small payloads only).
        mime_type (str | None): MIME type of the media.
        size_bytes (int | None): Size in bytes.
        sha256 (str | None): SHA256 hash of the media.
        filename (str | None): Filename of the media.
        width (int | None): Image width (if applicable).
        height (int | None): Image height (if applicable).
        duration_ms (int | None): Duration in milliseconds (if applicable).
        page (int | None): Page number (if applicable).
    """

    kind: Literal["url", "file_id", "data"] = "url"
    url: str | None = None  # http(s) or data: URL
    file_id: str | None = None  # provider-managed ID (e.g., OpenAI/Gemini)
    data_base64: str | None = None  # small payloads only
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    filename: str | None = None
    # Media-specific hints
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    page: int | None = None


class AnnotationRef(BaseModel):
    """
    Reference to annotation metadata (e.g., citation, note).

    Attributes:
        url (str | None): URL to annotation source.
        file_id (str | None): Provider-managed file ID.
        page (int | None): Page number (if applicable).
        index (int | None): Index within the annotation source.
        title (str | None): Title of the annotation.
    """

    url: str | None = None
    file_id: str | None = None
    page: int | None = None
    index: int | None = None
    title: str | None = None


class TextBlock(BaseModel):
    """
    Text content block for messages.

    Attributes:
        type (Literal["text"]): Block type discriminator.
        text (str): Text content.
        annotations (list[AnnotationRef]): List of annotation references.
    """

    type: Literal["text"] = "text"
    text: str
    annotations: list[AnnotationRef] = Field(default_factory=list)


class ImageBlock(BaseModel):
    """
    Image content block for messages.

    Attributes:
        type (Literal["image"]): Block type discriminator.
        media (MediaRef): Reference to image media.
        alt_text (str | None): Alternative text for accessibility.
        bbox (list[float] | None): Bounding box coordinates [x1, y1, x2, y2].
    """

    type: Literal["image"] = "image"
    media: MediaRef
    alt_text: str | None = None
    bbox: list[float] | None = None  # [x1,y1,x2,y2] if applicable


class AudioBlock(BaseModel):
    """
    Audio content block for messages.

    Attributes:
        type (Literal["audio"]): Block type discriminator.
        media (MediaRef): Reference to audio media.
        transcript (str | None): Transcript of audio.
        sample_rate (int | None): Sample rate in Hz.
        channels (int | None): Number of audio channels.
    """

    type: Literal["audio"] = "audio"
    media: MediaRef
    transcript: str | None = None
    sample_rate: int | None = None
    channels: int | None = None


class VideoBlock(BaseModel):
    """
    Video content block for messages.

    Attributes:
        type (Literal["video"]): Block type discriminator.
        media (MediaRef): Reference to video media.
        thumbnail (MediaRef | None): Reference to thumbnail image.
    """

    type: Literal["video"] = "video"
    media: MediaRef
    thumbnail: MediaRef | None = None


class DocumentBlock(BaseModel):
    """
    Document content block for messages.

    Attributes:
        type (Literal["document"]): Block type discriminator.
        media (MediaRef): Reference to document media.
        pages (list[int] | None): List of page numbers.
        excerpt (str | None): Excerpt from the document.
    """

    type: Literal["document"] = "document"
    media: MediaRef
    pages: list[int] | None = None
    excerpt: str | None = None


class DataBlock(BaseModel):
    """
    Data content block for messages.

    Attributes:
        type (Literal["data"]): Block type discriminator.
        mime_type (str): MIME type of the data.
        data_base64 (str | None): Base64-encoded data.
        media (MediaRef | None): Reference to associated media.
    """

    type: Literal["data"] = "data"
    mime_type: str
    data_base64: str | None = None
    media: MediaRef | None = None


class ToolCallBlock(BaseModel):
    """
    Tool call content block for messages.

    Attributes:
        type (Literal["tool_call"]): Block type discriminator.
        id (str): Tool call ID.
        name (str): Tool name.
        args (dict[str, Any]): Arguments for the tool call.
        tool_type (str | None): Type of tool (e.g., web_search, file_search).
    """

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    tool_type: str | None = None  # e.g., web_search, file_search, computer_use


class ToolResultBlock(BaseModel):
    """
    Tool result content block for messages.

    Attributes:
        type (Literal["tool_result"]): Block type discriminator.
        call_id (str): Tool call ID.
        output (Any): Output from the tool (str, dict, MediaRef, or list of blocks).
        is_error (bool): Whether the result is an error.
        status (Literal["completed", "failed"] | None): Status of the tool call.
    """

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    output: Any = None  # str | dict | MediaRef | list[ContentBlock-like]
    is_error: bool = False
    status: Literal["completed", "failed"] | None = None


class ReasoningBlock(BaseModel):
    """
    Reasoning content block for messages.

    Attributes:
        type (Literal["reasoning"]): Block type discriminator.
        summary (str): Summary of reasoning.
        details (list[str] | None): Detailed reasoning steps.
    """

    type: Literal["reasoning"] = "reasoning"
    summary: str
    details: list[str] | None = None


class AnnotationBlock(BaseModel):
    """
    Annotation content block for messages.

    Attributes:
        type (Literal["annotation"]): Block type discriminator.
        kind (Literal["citation", "note"]): Kind of annotation.
        refs (list[AnnotationRef]): List of annotation references.
        spans (list[tuple[int, int]] | None): Spans covered by the annotation.
    """

    type: Literal["annotation"] = "annotation"
    kind: Literal["citation", "note"] = "citation"
    refs: list[AnnotationRef] = Field(default_factory=list)
    spans: list[tuple[int, int]] | None = None


class ErrorBlock(BaseModel):
    """
    Error content block for messages.

    Attributes:
        type (Literal["error"]): Block type discriminator.
        message (str): Error message.
        code (str | None): Error code.
        data (dict[str, Any] | None): Additional error data.
    """

    type: Literal["error"] = "error"
    message: str
    code: str | None = None
    data: dict[str, Any] | None = None


# Discriminated union over the "type" field
ContentBlock = Annotated[
    Union[
        TextBlock,
        ImageBlock,
        AudioBlock,
        VideoBlock,
        DocumentBlock,
        DataBlock,
        ToolCallBlock,
        ToolResultBlock,
        ReasoningBlock,
        AnnotationBlock,
        ErrorBlock,
    ],
    Field(discriminator="type"),
]


class Message(BaseModel):
    """
    Represents a message in a conversation, including content, role, metadata, and token usage.

    Attributes:
        message_id (str | int): Unique identifier for the message.
        role (Literal["user", "assistant", "system", "tool"]): The role of the message sender.
        content (list[ContentBlock]): The message content blocks.
        delta (bool): Indicates if this is a delta/partial message.
        tools_calls (list[dict[str, Any]] | None): Tool call information, if any.
        reasoning (str | None): Reasoning or explanation, if any.
        timestamp (datetime | None): Timestamp of the message.
        metadata (dict[str, Any]): Additional metadata.
        usages (TokenUsages | None): Token usage statistics.
        raw (dict[str, Any] | None): Raw data, if any.

    Example:
        >>> msg = Message(message_id="abc123", role="user", content=[TextBlock(text="Hello!")])
        {'message_id': 'abc123', 'role': 'user', 'content': [...], ...}
    """

    message_id: str | int = Field(default_factory=lambda: generate_id(None))
    role: Literal["user", "assistant", "system", "tool"]
    content: list[ContentBlock]
    delta: bool = False  # Indicates if this is a delta/partial message
    tools_calls: list[dict[str, Any]] | None = None
    reasoning: str | None = None  # Remove it
    timestamp: datetime | None = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    usages: TokenUsages | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def text_message(
        cls,
        content: str,
        role: Literal["user", "assistant", "system", "tool"] = "user",
        message_id: str | None = None,
    ) -> "Message":
        """
        Create a Message instance from plain text.

        Args:
            content (str): The message content.
            role (Literal["user", "assistant", "system", "tool"]): The role of the sender.
            message_id (str | None): Optional message ID.

        Returns:
            Message: The created Message instance.

        Example:
            >>> Message.text_message("Hello!", role="user")
        """
        logger.debug("Creating message from text with role: %s", role)
        return cls(
            message_id=generate_id(message_id),
            role=role,
            content=[TextBlock(text=content)],
            timestamp=datetime.now(),
            metadata={},
        )

    @classmethod
    def tool_message(
        cls,
        content: list[ContentBlock],
        message_id: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "Message":
        """
        Create a tool message, optionally marking it as an error.

        Args:
            content (list[ContentBlock]): The message content blocks.
            message_id (str | None): Optional message ID.
            meta (dict[str, Any] | None): Optional metadata.

        Returns:
            Message: The created tool message instance.

        Example:
            >>> Message.tool_message([ToolResultBlock(...)], message_id="tool1")
        """
        res = content
        msg_id = generate_id(message_id)
        return cls(
            message_id=msg_id,
            role="tool",
            content=res,
            timestamp=datetime.now(),
            metadata=meta or {},
        )

    # --- Convenience helpers ---
    def text(self) -> str:
        """
        Best-effort text extraction from content blocks.

        Returns:
            str: Concatenated text from TextBlock and ToolResultBlock outputs.

        Example:
            >>> msg.text()
            'Hello!Result text.'
        """
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ToolResultBlock) and isinstance(block.output, str):
                parts.append(block.output)
        return "".join(parts)

    def attach_media(
        self,
        media: MediaRef,
        as_type: Literal["image", "audio", "video", "document"],
    ) -> None:
        """
        Append a media block to the content.

        If content was text, creates a block list. Supports image, audio, video, and document types.

        Args:
            media (MediaRef): Reference to media content.
            as_type (Literal["image", "audio", "video", "document"]): Type of media block to append.

        Returns:
            None

        Raises:
            ValueError: If an unsupported media type is provided.

        Example:
            >>> msg.attach_media(media_ref, as_type="image")
        """
        block: ContentBlock
        if as_type == "image":
            block = ImageBlock(media=media)
        elif as_type == "audio":
            block = AudioBlock(media=media)
        elif as_type == "video":
            block = VideoBlock(media=media)
        elif as_type == "document":
            block = DocumentBlock(media=media)
        else:
            raise ValueError(f"Unsupported media type: {as_type}")

        if isinstance(self.content, str):
            self.content = [TextBlock(text=self.content), block]
        elif isinstance(self.content, list):
            self.content.append(block)
        else:
            self.content = [block]
