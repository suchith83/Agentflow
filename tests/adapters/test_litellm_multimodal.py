"""Tests for LiteLLM converter multimodal support (audio, images, reasoning)."""
import pytest
from unittest.mock import Mock, patch

from agentflow.adapters.llm.litellm_converter import LiteLLMConverter
from agentflow.state.message_block import AudioBlock, ImageBlock, ReasoningBlock, TextBlock


class MockModelResponse:
    """Mock ModelResponse for testing."""

    def __init__(self, data):
        self.id = data.get("id", "test_id")
        self._data = data

    def model_dump(self):
        return self._data


class MockModelResponseStream:
    """Mock ModelResponseStream for testing."""

    def __init__(self, id="test_id", choices=None):
        self.id = id
        self.choices = choices or []


class MockDelta:
    """Mock delta for streaming response."""

    def __init__(
        self, content="", reasoning_content="", audio=None, images=None, tool_calls=None
    ):
        self.content = content
        self.reasoning_content = reasoning_content
        self.audio = audio
        self.images = images
        self.tool_calls = tool_calls or []


class MockChoice:
    """Mock choice for streaming response."""

    def __init__(self, delta=None):
        self.delta = delta


class TestLiteLLMMultimodalConverter:
    """Test class for LiteLLM converter multimodal features."""

    @pytest.fixture
    def converter(self):
        """Provide LiteLLMConverter instance."""
        return LiteLLMConverter()

    @patch("agentflow.adapters.llm.litellm_converter.HAS_LITELLM", True)
    @pytest.mark.asyncio
    async def test_convert_response_with_audio(self, converter):
        """Test response conversion with audio content."""
        response_data = {
            "id": "audio_test",
            "choices": [
                {
                    "message": {
                        "content": "Audio response",
                        "audio": {
                            "id": "audio_123",
                            "data": "base64encodeddata",
                            "transcript": "Hello world",
                            "expires_at": 1234567890,
                        },
                    }
                }
            ],
            "usage": {},
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        # Check that we have both text and audio blocks
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Audio response"
        assert isinstance(message.content[1], AudioBlock)
        assert message.content[1].media.data_base64 == "base64encodeddata"
        assert message.content[1].transcript == "Hello world"

    @patch("agentflow.adapters.llm.litellm_converter.HAS_LITELLM", True)
    @pytest.mark.asyncio
    async def test_convert_response_with_images(self, converter):
        """Test response conversion with image content."""
        response_data = {
            "id": "image_test",
            "choices": [
                {
                    "message": {
                        "content": "Image response",
                        "images": [
                            {"url": "https://example.com/image1.jpg"},
                            {"url": "https://example.com/image2.jpg"},
                        ],
                    }
                }
            ],
            "usage": {},
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        # Check that we have text and image blocks
        assert len(message.content) == 3
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ImageBlock)
        assert message.content[1].media.url == "https://example.com/image1.jpg"
        assert isinstance(message.content[2], ImageBlock)
        assert message.content[2].media.url == "https://example.com/image2.jpg"

    @patch("agentflow.adapters.llm.litellm_converter.HAS_LITELLM", True)
    @pytest.mark.asyncio
    async def test_convert_response_with_reasoning(self, converter):
        """Test response conversion with reasoning content."""
        response_data = {
            "id": "reasoning_test",
            "choices": [
                {
                    "message": {
                        "content": "Text response",
                        "reasoning_content": "I thought about this carefully",
                    }
                }
            ],
            "usage": {},
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        # Check that we have both text and reasoning blocks
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Text response"
        assert isinstance(message.content[1], ReasoningBlock)
        assert message.content[1].summary == "I thought about this carefully"
        assert message.reasoning == "I thought about this carefully"

    @patch("agentflow.adapters.llm.litellm_converter.HAS_LITELLM", True)
    @pytest.mark.asyncio
    async def test_convert_response_multimodal_combined(self, converter):
        """Test response conversion with all multimodal types combined."""
        response_data = {
            "id": "multimodal_test",
            "choices": [
                {
                    "message": {
                        "content": "Combined response",
                        "reasoning_content": "Reasoning about the content",
                        "audio": {
                            "id": "audio_123",
                            "data": "audiodata",
                            "transcript": "Audio transcript",
                        },
                        "images": [{"url": "https://example.com/img.jpg"}],
                    }
                }
            ],
            "usage": {},
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        # Check all block types are present
        assert len(message.content) == 4
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ReasoningBlock)
        assert isinstance(message.content[2], AudioBlock)
        assert isinstance(message.content[3], ImageBlock)

    @pytest.mark.asyncio
    async def test_process_chunk_with_audio(self, converter):
        """Test _process_chunk handles audio in delta."""
        audio_data = {
            "id": "audio_456",
            "data": "streamaudiodata",
            "transcript": "Streaming audio",
        }
        delta = MockDelta(content="Text", audio=audio_data)
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("audio_chunk", [choice])

        result = converter._process_chunk(chunk, 1, "", "", [], set())

        accumulated_content, accumulated_reasoning, tool_calls, seq, message = result
        assert accumulated_content == "Text"
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], AudioBlock)

    @pytest.mark.asyncio
    async def test_process_chunk_with_images(self, converter):
        """Test _process_chunk handles images in delta."""
        images_data = [{"url": "https://example.com/stream.jpg"}]
        delta = MockDelta(content="Text", images=images_data)
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("image_chunk", [choice])

        result = converter._process_chunk(chunk, 1, "", "", [], set())

        accumulated_content, accumulated_reasoning, tool_calls, seq, message = result
        assert accumulated_content == "Text"
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ImageBlock)

    @pytest.mark.asyncio
    async def test_process_chunk_with_reasoning(self, converter):
        """Test _process_chunk handles reasoning in delta."""
        delta = MockDelta(content="Text", reasoning_content="Thinking...")
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("reasoning_chunk", [choice])

        result = converter._process_chunk(chunk, 1, "", "", [], set())

        accumulated_content, accumulated_reasoning, tool_calls, seq, message = result
        assert accumulated_content == "Text"
        assert accumulated_reasoning == "Thinking..."
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ReasoningBlock)

    def test_extract_audio_block_valid(self, converter):
        """Test _extract_audio_block with valid audio data."""
        audio_data = {
            "id": "audio_789",
            "data": "validaudiodata",
            "transcript": "Valid transcript",
        }

        block = converter._extract_audio_block(audio_data)

        assert block is not None
        assert isinstance(block, AudioBlock)
        assert block.media.data_base64 == "validaudiodata"
        assert block.transcript == "Valid transcript"

    def test_extract_audio_block_missing_data(self, converter):
        """Test _extract_audio_block with missing data."""
        audio_data = {"id": "audio_empty", "transcript": "No data"}

        block = converter._extract_audio_block(audio_data)

        assert block is None

    def test_extract_audio_block_invalid_format(self, converter):
        """Test _extract_audio_block with invalid format."""
        audio_data = "invalid_format"

        block = converter._extract_audio_block(audio_data)

        assert block is None

    def test_extract_image_blocks_valid(self, converter):
        """Test _extract_image_blocks with valid image data."""
        images_data = [
            {"url": "https://example.com/img1.jpg"},
            {"url": "https://example.com/img2.jpg"},
        ]

        blocks = converter._extract_image_blocks(images_data)

        assert len(blocks) == 2
        assert all(isinstance(b, ImageBlock) for b in blocks)
        assert blocks[0].media.url == "https://example.com/img1.jpg"
        assert blocks[1].media.url == "https://example.com/img2.jpg"

    def test_extract_image_blocks_missing_url(self, converter):
        """Test _extract_image_blocks with missing URLs."""
        images_data = [{"url": "https://example.com/valid.jpg"}, {"no_url": "invalid"}]

        blocks = converter._extract_image_blocks(images_data)

        # Should only extract the valid one
        assert len(blocks) == 1
        assert blocks[0].media.url == "https://example.com/valid.jpg"

    def test_extract_image_blocks_invalid_format(self, converter):
        """Test _extract_image_blocks with invalid format."""
        images_data = "invalid_format"

        blocks = converter._extract_image_blocks(images_data)

        assert len(blocks) == 0

    @patch("agentflow.adapters.llm.litellm_converter.HAS_LITELLM", True)
    @pytest.mark.asyncio
    async def test_convert_response_empty_content_with_audio(self, converter):
        """Test response conversion with empty content but audio present."""
        response_data = {
            "id": "audio_only_test",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "audio": {
                            "id": "audio_only",
                            "data": "onlyaudiodata",
                            "transcript": "Only audio",
                        },
                    }
                }
            ],
            "usage": {},
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        # Should only have audio block
        assert len(message.content) == 1
        assert isinstance(message.content[0], AudioBlock)

    @patch("agentflow.adapters.llm.litellm_converter.HAS_LITELLM", True)
    @pytest.mark.asyncio
    async def test_convert_response_empty_reasoning(self, converter):
        """Test that empty reasoning_content is handled correctly."""
        response_data = {
            "id": "empty_reasoning_test",
            "choices": [{"message": {"content": "Text", "reasoning_content": ""}}],
            "usage": {},
        }

        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)

        # Should only have text block, no reasoning
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextBlock)
        assert message.reasoning == ""
