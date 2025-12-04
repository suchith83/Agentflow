"""
Basic tests for OpenAI converter functionality.
"""

import json
from unittest.mock import Mock, patch
from typing import Any

import pytest

from agentflow.adapters.llm.openai_converter import OpenAIConverter
from agentflow.state.message import Message, TokenUsages
from agentflow.state.message_block import (
    AudioBlock,
    ImageBlock,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
)


class MockModelResponse:
    """Mock ChatCompletion response for testing."""

    def __init__(self, data):
        self.id = data.get("id", "test_id")
        self.model = data.get("model", "gpt-4o")
        self.created = data.get("created", 1234567890)
        
        # Mock usage
        usage_data = data.get("usage", {})
        self.usage = type('Usage', (), {
            'prompt_tokens': usage_data.get('prompt_tokens', 10),
            'completion_tokens': usage_data.get('completion_tokens', 20),
            'total_tokens': usage_data.get('total_tokens', 30)
        })
        
        # Mock choices
        choices_data = data.get("choices", [{}])
        self.choices = []
        for choice_data in choices_data:
            message_data = choice_data.get("message", {})
            message = type('Message', (), {
                'role': message_data.get('role', 'assistant'),
                'content': message_data.get('content'),
                'audio': message_data.get('audio'),
                'images': message_data.get('images'),
                'reasoning_content': message_data.get('reasoning_content'),
                'tool_calls': message_data.get('tool_calls')
            })
            choice = type('Choice', (), {
                'message': message,
                'finish_reason': choice_data.get('finish_reason', 'stop')
            })
            self.choices.append(choice)


class TestOpenAIConverter:
    """Test class for OpenAI converter."""

    @pytest.fixture
    def converter(self):
        """Provide OpenAIConverter instance."""
        return OpenAIConverter()

    @patch("agentflow.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_basic_text_conversion(self, converter):
        """Test basic text message conversion."""
        response_data = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello, world!",
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert isinstance(message, Message)
        assert message.role == "assistant"
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Hello, world!"

    @patch("agentflow.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_audio_conversion(self, converter):
        """Test audio content conversion."""
        response_data = {
            "id": "chatcmpl-audio",
            "choices": [
                {
                    "message": {
                        "content": "Audio response",
                        "audio": {
                            "id": "audio_123",
                            "data": "base64encodeddata",
                            "transcript": "Hello from audio",
                        }
                    }
                }
            ],
            "usage": {}
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Audio response"
        assert isinstance(message.content[1], AudioBlock)
        assert message.content[1].transcript == "Hello from audio"
        assert message.content[1].media.data_base64 == "base64encodeddata"

    @patch("agentflow.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_images_conversion(self, converter):
        """Test image content conversion."""
        response_data = {
            "id": "chatcmpl-images",
            "choices": [
                {
                    "message": {
                        "content": "Here are images",
                        "images": [
                            {"url": "https://example.com/image1.png"},
                            {"url": "https://example.com/image2.png"}
                        ]
                    }
                }
            ],
            "usage": {}
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        # Should have 1 text block + 2 image blocks
        assert len(message.content) == 3
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ImageBlock)
        assert isinstance(message.content[2], ImageBlock)
        assert message.content[1].media.url == "https://example.com/image1.png"
        assert message.content[2].media.url == "https://example.com/image2.png"

    @patch("agentflow.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_tool_calls_conversion(self, converter):
        """Test tool call conversion."""
        response_data = {
            "id": "chatcmpl-tools",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            type('ToolCall', (), {
                                'id': 'call_123',
                                'type': 'function',
                                'function': type('Function', (), {
                                    'name': 'get_weather',
                                    'arguments': json.dumps({"location": "SF"})
                                })
                            })
                        ]
                    }
                }
            ],
            "usage": {}
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert len(message.content) == 1
        assert isinstance(message.content[0], ToolCallBlock)
        assert message.content[0].name == "get_weather"
        assert message.content[0].args == {"location": "SF"}
        assert message.content[0].id == "call_123"

    @patch("agentflow.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_empty_content(self, converter):
        """Test handling of empty/null content."""
        response_data = {
            "id": "chatcmpl-empty",
            "choices": [
                {
                    "message": {
                        "content": None,
                    }
                }
            ],
            "usage": {}
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert isinstance(message, Message)
        assert len(message.content) == 0

    @patch("agentflow.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_token_usage(self, converter):
        """Test token usage extraction."""
        response_data = {
            "id": "chatcmpl-usage",
            "choices": [
                {
                    "message": {
                        "content": "Test",
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert isinstance(message.usages, TokenUsages)
        assert message.usages.prompt_tokens == 100
        assert message.usages.completion_tokens == 50
        assert message.usages.total_tokens == 150

    @patch("agentflow.adapters.llm.openai_converter.HAS_OPENAI", True)
    @pytest.mark.asyncio
    async def test_metadata_extraction(self, converter):
        """Test metadata extraction."""
        response_data = {
            "id": "chatcmpl-meta",
            "model": "gpt-4o-2024-05-13",
            "choices": [
                {
                    "message": {
                        "content": "Test",
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {}
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        assert message.metadata["provider"] == "openai"
        assert message.metadata["model"] == "gpt-4o-2024-05-13"
        assert message.metadata["finish_reason"] == "stop"
