"""
Integration adapters for optional third-party LLM SDKs.

This module exposes universal converter APIs to normalize responses and
streaming outputs from popular LLM SDKs (e.g., LiteLLM, OpenAI) for use in
agentflow agent graphs. Adapters provide unified conversion, streaming,
and message normalization for agent workflows.

Exports:
    BaseConverter: Abstract base class for LLM response converters.
    ConverterType: Enum of supported converter types.
    LiteLLMConverter: Converter for LiteLLM responses and streams.
    # OpenAIConverter: (commented, available if implemented)
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
