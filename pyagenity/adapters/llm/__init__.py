"""Integration adapters for optional third-party SDKs.

Includes a universal converter API to normalize responses/streams from
popular SDKs such as LiteLLM and OpenAI.
"""

from .base_converter import BaseConverter, ConverterType
from .litellm_converter import LiteLLMConverter


# from .openai_converter import OpenAIConverter


__all__ = [
    "BaseConverter",
    "ConverterType",
    "LiteLLMConverter",
    # "OpenAIConverter",
]
