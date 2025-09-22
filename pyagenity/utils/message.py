# Default message representation
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
    """Reference to media content (image/audio/video/document/data).

    Prefer referencing by URL or provider file_id over inlining base64 for large payloads.
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
    url: str | None = None
    file_id: str | None = None
    page: int | None = None
    index: int | None = None
    title: str | None = None


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    annotations: list[AnnotationRef] = Field(default_factory=list)


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    media: MediaRef
    alt_text: str | None = None
    bbox: list[float] | None = None  # [x1,y1,x2,y2] if applicable


class AudioBlock(BaseModel):
    type: Literal["audio"] = "audio"
    media: MediaRef
    transcript: str | None = None
    sample_rate: int | None = None
    channels: int | None = None


class VideoBlock(BaseModel):
    type: Literal["video"] = "video"
    media: MediaRef
    thumbnail: MediaRef | None = None


class DocumentBlock(BaseModel):
    type: Literal["document"] = "document"
    media: MediaRef
    pages: list[int] | None = None
    excerpt: str | None = None


class DataBlock(BaseModel):
    type: Literal["data"] = "data"
    mime_type: str
    data_base64: str | None = None
    media: MediaRef | None = None


class ToolCallBlock(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    tool_type: str | None = None  # e.g., web_search, file_search, computer_use


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    output: Any = None  # str | dict | MediaRef | list[ContentBlock-like]
    is_error: bool = False
    status: Literal["completed", "failed"] | None = None


class ReasoningBlock(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    summary: str
    details: list[str] | None = None


class AnnotationBlock(BaseModel):
    type: Literal["annotation"] = "annotation"
    kind: Literal["citation", "note"] = "citation"
    refs: list[AnnotationRef] = Field(default_factory=list)
    spans: list[tuple[int, int]] | None = None


class ErrorBlock(BaseModel):
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

    Example:
        >>> msg = Message(message_id="abc123", role="user", content="Hello!")
        {'message_id': 'abc123', 'role': 'user', 'content': 'Hello!', ...}

    Attributes:
        message_id (str): Unique identifier for the message.
        role (Literal["user", "assistant", "system", "tool"]): The role of the message sender.
        content (str): The message content.
        tools_calls (list[dict[str, Any]] | None): Tool call information, if any.
        function_call (dict[str, Any] | None): Function call information, if any.
        reasoning (str | None): Reasoning or explanation, if any.
        timestamp (datetime | None): Timestamp of the message.
        metadata (dict[str, Any]): Additional metadata.
        usages (TokenUsages | None): Token usage statistics.
        raw (dict[str, Any] | None): Raw data, if any.
    """

    message_id: str | int = Field(default_factory=lambda: generate_id(None))
    role: Literal["user", "assistant", "system", "tool"]
    content: list[ContentBlock]
    delta: bool = False  # Indicates if this is a delta/partial message
    tools_calls: list[dict[str, Any]] | None = None
    reasoning: str | None = None
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
            data (str): The message content.
            role (Literal["user", "assistant", "system", "tool"]): The role of the sender.
            message_id (str | None): Optional message ID.

        Returns:
            Message: The created Message instance.
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
    ):
        """
        Create a tool message, optionally marking it as an error.

        Args:
            content (str): The message content.
            is_error (bool): Whether this message represents an error.
            message_id (str | None): Optional message ID.
            meta (dict[str, Any] | None): Optional metadata.

        Returns:
            Message: The created tool message instance.
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
        """Best-effort text extraction from content.

        - If content is a string, return it.
        - If it's a list of blocks, join text fields and tool_result string outputs.
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
        """Append a media block to the content (creates block list if content was text)."""
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
