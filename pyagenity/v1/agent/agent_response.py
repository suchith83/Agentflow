from dataclasses import dataclass
from typing import Any


@dataclass
class AgentResponseChunk:
    """Represents a streaming delta chunk from the model."""

    delta: str = ""
    done: bool = False

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"AgentResponseChunk(delta={self.delta!r}, done={self.done})"


@dataclass
class AgentResponse:
    content: str = ""
    thinking: Any | None = None
    usage: dict | None = None
    raw: str | None = None
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "thinking": self.thinking,
            "usage": self.usage,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
        }
