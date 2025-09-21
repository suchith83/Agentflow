from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from pyagenity.publisher.events import EventModel
from pyagenity.state.agent_state import AgentState
from pyagenity.utils.message import Message


class ConverterType(Enum):
    OPENAI = "openai"
    LITELLM = "litellm"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"


class BaseConverter(ABC):
    """Base class for all response converters"""

    def __init__(self, state: AgentState | None = None) -> None:
        self.state = state

    @abstractmethod
    async def convert_response(self, response: Any) -> Message:
        """Convert the agent response to the target format"""
        raise NotImplementedError("Conversion not implemented for this converter")

    @abstractmethod
    async def convert_streaming_response(
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[EventModel | Message, None]:
        """Convert streaming response to the target format.

        Implementations should return an async generator (defined with
        `async def ...` and `yield`). Declaring this as a regular abstract
        method ensures type compatibility across subclasses that implement
        async generators.
        """
        raise NotImplementedError("Streaming not implemented for this converter")
