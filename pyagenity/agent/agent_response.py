from typing import Any, Optional, Dict
from dataclasses import dataclass


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
    thinking: Optional[Any] = None
    usage: Optional[dict] = None
    raw: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "thinking": self.thinking,
            "usage": self.usage,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
        }
