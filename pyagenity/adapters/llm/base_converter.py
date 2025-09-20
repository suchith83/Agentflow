from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pyagenity.utils.message import Message
from pyagenity.utils.streaming import EventModel


class ConverterType(Enum):
    OPENAI = "openai"
    LITELLM = "litellm"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CUSTOM = "custom"


@dataclass
class ConverterConfig:
    """Configuration for the converter"""

    converter_type: ConverterType
    stream: bool = False
    custom_params: dict[str, Any] | None = None


class BaseConverter(ABC):
    """Base class for all response converters"""

    def __init__(self, config: ConverterConfig):
        self.config = config

    @abstractmethod
    async def convert_response(self, response: Any, **kwargs) -> Message:
        """Convert the agent response to the target format"""
        raise NotImplementedError("Conversion not implemented for this converter")

    @abstractmethod
    async def convert_streaming_response(
        self, response: Any, **kwargs
    ) -> AsyncGenerator[EventModel]:
        """Convert streaming response to the target format"""
        raise NotImplementedError("Streaming not implemented for this converter")
