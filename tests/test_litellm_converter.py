"""Tests for LiteLLMConverter with mock responses."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pyagenity.utils.message import Message, TokenUsages, TextBlock, ReasoningBlock, ToolCallBlock


class TestLiteLLMConverter:
    """Test suite for LiteLLMConverter."""

    def setup_method(self):
        """Set up test environment with mocked LiteLLM."""
        # Mock the entire litellm module
        self.mock_litellm = MagicMock()
        self.mock_model_response = MagicMock()
        self.mock_model_response_stream = MagicMock()
        self.mock_custom_stream_wrapper = MagicMock()
        
        # Create mock classes that the converter expects
        self.mock_litellm.ModelResponse = self.mock_model_response
        self.mock_litellm.types = MagicMock()
        self.mock_litellm.types.utils = MagicMock()
        self.mock_litellm.types.utils.ModelResponse = self.mock_model_response
        self.mock_litellm.types.utils.ModelResponseStream = self.mock_model_response_stream
        self.mock_litellm.CustomStreamWrapper = self.mock_custom_stream_wrapper

    def create_mock_response(self, content="Test response", tool_calls=None, reasoning_content=""):
        """Create a mock LiteLLM response."""
        mock_response = Mock()
        mock_response.id = "test_id_123"
        mock_response.model_dump.return_value = {
            "id": "test_id_123",
            "choices": [{
                "message": {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls or []
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "completion_tokens": 10,
                "prompt_tokens": 20,
                "total_tokens": 30,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "prompt_tokens_details": {"reasoning_tokens": 5},
                "completion_tokens_details": {}
            },
            "model": "gpt-3.5-turbo",
            "object": "chat.completion",
            "created": datetime.now()
        }
        return mock_response

    def test_import_without_litellm(self):
        """Test that converter fails gracefully when LiteLLM is not available."""
        with patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', False):
            # This should still import but HAS_LITELLM should be False
            from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
            converter = LiteLLMConverter()
            
            # But conversion should fail
            with pytest.raises(ImportError, match="litellm is not installed"):
                asyncio.run(converter.convert_response(Mock()))

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    def test_convert_response_basic(self):
        """Test basic response conversion."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        mock_response = self.create_mock_response("Hello world")
        
        result = asyncio.run(converter.convert_response(mock_response))
        
        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert result.text() == "Hello world"
        assert result.usages is not None
        assert result.usages.completion_tokens == 10
        assert result.usages.prompt_tokens == 20
        assert result.usages.total_tokens == 30
        assert result.metadata["model"] == "gpt-3.5-turbo"
        assert result.metadata["finish_reason"] == "stop"

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    def test_convert_response_with_reasoning(self):
        """Test response conversion with reasoning content."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        mock_response = self.create_mock_response(
            content="Final answer",
            reasoning_content="I need to think about this..."
        )
        
        result = asyncio.run(converter.convert_response(mock_response))
        
        assert result.text() == "Final answer"
        assert result.reasoning == "I need to think about this..."
        
        # Check that reasoning block is in content
        reasoning_blocks = [block for block in result.content if isinstance(block, ReasoningBlock)]
        assert len(reasoning_blocks) == 1
        assert reasoning_blocks[0].summary == "I need to think about this..."

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    def test_convert_response_with_tool_calls(self):
        """Test response conversion with tool calls."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        tool_calls = [{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "New York"}'
            }
        }]
        
        converter = LiteLLMConverter()
        mock_response = self.create_mock_response(
            content="I'll check the weather",
            tool_calls=tool_calls
        )
        
        result = asyncio.run(converter.convert_response(mock_response))
        
        assert result.text() == "I'll check the weather"
        assert result.tools_calls is not None
        assert len(result.tools_calls) == 1

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    def test_convert_response_empty_content(self):
        """Test response conversion with empty content."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        mock_response = self.create_mock_response(content="")
        
        result = asyncio.run(converter.convert_response(mock_response))
        
        assert result.text() == ""
        assert len(result.content) == 0  # No text blocks for empty content

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    def test_convert_response_null_content(self):
        """Test response conversion with null content."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        mock_response = Mock()
        mock_response.id = "test_id"
        mock_response.model_dump.return_value = {
            "id": "test_id",
            "choices": [{
                "message": {
                    "content": None,  # None content
                    "reasoning_content": "",
                    "tool_calls": []
                },
                "finish_reason": "stop"
            }],
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
            "model": "gpt-3.5-turbo",
            "object": "chat.completion",
            "created": datetime.now()
        }
        
        result = asyncio.run(converter.convert_response(mock_response))
        
        assert result.text() == ""

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)  
    def test_convert_response_missing_usage(self):
        """Test response conversion when usage data is missing."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        mock_response = Mock()
        mock_response.id = "test_id"
        mock_response.model_dump.return_value = {
            "id": "test_id",
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}],
            "usage": {},  # Empty usage
            "model": "gpt-3.5-turbo",
            "object": "chat.completion",
            "created": datetime.now()
        }
        
        converter = LiteLLMConverter()
        result = asyncio.run(converter.convert_response(mock_response))
        
        assert result.usages is not None
        assert result.usages.completion_tokens == 0
        assert result.usages.prompt_tokens == 0
        assert result.usages.total_tokens == 0

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    def test_process_chunk_basic(self):
        """Test _process_chunk method with basic chunk."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        
        # Create mock chunk
        mock_chunk = Mock()
        mock_chunk.id = "chunk_id"
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = "Hello"
        mock_chunk.choices[0].delta.reasoning_content = None
        mock_chunk.choices[0].delta.tool_calls = None
        
        result = converter._process_chunk(
            chunk=mock_chunk,
            seq=0,
            accumulated_content="",
            accumulated_reasoning_content="",
            tool_calls=[],
            tool_ids=set()
        )
        
        # Should return updated content and message
        updated_content, updated_reasoning, updated_tools, seq, message = result
        assert updated_content == "Hello"
        assert message is not None
        assert message.text() == "Hello"

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    def test_process_chunk_with_reasoning(self):
        """Test _process_chunk with reasoning content."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        
        mock_chunk = Mock()
        mock_chunk.id = "chunk_id"
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = ""
        mock_chunk.choices[0].delta.reasoning_content = "Thinking..."
        mock_chunk.choices[0].delta.tool_calls = None
        
        result = converter._process_chunk(
            chunk=mock_chunk,
            seq=0,
            accumulated_content="",
            accumulated_reasoning_content="",
            tool_calls=[],
            tool_ids=set()
        )
        
        updated_content, updated_reasoning, updated_tools, seq, message = result
        assert updated_reasoning == "Thinking..."

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    def test_process_chunk_none(self):
        """Test _process_chunk with None chunk."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        
        result = converter._process_chunk(
            chunk=None,
            seq=0,
            accumulated_content="existing",
            accumulated_reasoning_content="",
            tool_calls=[],
            tool_ids=set()
        )
        
        # Should return unchanged values
        updated_content, updated_reasoning, updated_tools, seq, message = result
        assert updated_content == "existing"
        assert message is None

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_handle_stream_async_iteration(self):
        """Test _handle_stream with async iteration."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        
        # Create mock stream that supports async iteration
        mock_stream = AsyncMock()
        
        # Create mock chunks
        chunk1 = Mock()
        chunk1.id = "chunk1"
        chunk1.choices = [Mock()]
        chunk1.choices[0].delta = Mock()
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.reasoning_content = None
        chunk1.choices[0].delta.tool_calls = None
        
        chunk2 = Mock()
        chunk2.id = "chunk2"
        chunk2.choices = [Mock()]
        chunk2.choices[0].delta = Mock()
        chunk2.choices[0].delta.content = " World"
        chunk2.choices[0].delta.reasoning_content = None
        chunk2.choices[0].delta.tool_calls = None
        
        # Mock async iteration
        mock_stream.__aiter__ = AsyncMock(return_value=iter([chunk1, chunk2]))
        
        results = []
        async for item in converter._handle_stream(
            config={"thread_id": "test"},
            node_name="test_node",
            stream=mock_stream
        ):
            results.append(item)
        
        # Should get final message with accumulated content
        assert len(results) >= 1
        final_message = results[-1]
        assert isinstance(final_message, Message)

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_streaming_response_with_stream_wrapper(self):
        """Test convert_streaming_response with CustomStreamWrapper."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        
        # Create mock stream wrapper
        mock_wrapper = Mock()
        
        # Mock the async generator
        async def mock_generator():
            yield Message.text_message("Stream result", "assistant")

        with patch.object(converter, '_handle_stream') as mock_handle_stream:
            mock_handle_stream.return_value = mock_generator()
            
            # Patch the CustomStreamWrapper import to match our mock
            with patch('pyagenity.adapters.llm.litellm_converter.CustomStreamWrapper', mock_wrapper.__class__):
                results = []
                async for item in converter.convert_streaming_response(
                    config={"thread_id": "test"},
                    node_name="test_node",
                    response=mock_wrapper
                ):
                    results.append(item)
                
                assert len(results) == 1
                assert results[0].text() == "Stream result"

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_streaming_response_with_model_response(self):
        """Test convert_streaming_response with regular ModelResponse."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        
        mock_response = self.create_mock_response("Direct response")
        
        # Patch the ModelResponse import to match our mock
        with patch('pyagenity.adapters.llm.litellm_converter.ModelResponse', mock_response.__class__):
            results = []
            async for item in converter.convert_streaming_response(
                config={"thread_id": "test"},
                node_name="test_node",
                response=mock_response
            ):
                results.append(item)
            
            assert len(results) == 1
            assert results[0].text() == "Direct response"

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_streaming_response_unsupported_type(self):
        """Test convert_streaming_response with unsupported response type."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        
        with pytest.raises(Exception):  # Should raise some exception for unsupported type
            async for item in converter.convert_streaming_response(
                config={"thread_id": "test"},
                node_name="test_node",
                response="unsupported_type"
            ):
                pass

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', False)
    @pytest.mark.asyncio
    async def test_convert_streaming_response_without_litellm(self):
        """Test convert_streaming_response fails without LiteLLM."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        
        with pytest.raises(ImportError, match="litellm is not installed"):
            async for item in converter.convert_streaming_response(
                config={},
                node_name="test",
                response=Mock()
            ):
                pass

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @patch('pyagenity.adapters.llm.litellm_converter.logger')
    def test_logging_during_conversion(self, mock_logger):
        """Test that appropriate logging occurs during conversion."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        mock_response = self.create_mock_response("Test message")
        
        asyncio.run(converter.convert_response(mock_response))
        
        # Should log the message creation
        mock_logger.debug.assert_called_with(
            "Creating message from model response with id: %s", 
            "test_id_123"
        )

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    def test_token_usage_parsing(self):
        """Test detailed token usage parsing."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        mock_response = Mock()
        mock_response.id = "test_id"
        mock_response.model_dump.return_value = {
            "id": "test_id",
            "choices": [{"message": {"content": "test"}, "finish_reason": "stop"}],
            "usage": {
                "completion_tokens": 25,
                "prompt_tokens": 50,
                "total_tokens": 75,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 10,
                "prompt_tokens_details": {"reasoning_tokens": 15},
                "completion_tokens_details": {"reasoning_tokens": 8}
            },
            "model": "gpt-4",
            "object": "chat.completion",
            "created": datetime.now()
        }
        
        converter = LiteLLMConverter()
        result = asyncio.run(converter.convert_response(mock_response))
        
        assert result.usages is not None
        assert result.usages.completion_tokens == 25
        assert result.usages.prompt_tokens == 50
        assert result.usages.total_tokens == 75
        assert result.usages.cache_creation_input_tokens == 5
        assert result.usages.cache_read_input_tokens == 10
        assert result.usages.reasoning_tokens == 15
        assert result.metadata["completion_tokens_details"] == {"reasoning_tokens": 8}

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    def test_metadata_extraction(self):
        """Test extraction of metadata fields."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        mock_response = Mock()
        mock_response.id = "test_id"
        mock_response.model_dump.return_value = {
            "id": "test_id",
            "choices": [{
                "message": {"content": "test"},
                "finish_reason": "length"
            }],
            "usage": {},
            "model": "gpt-4-turbo",
            "object": "chat.completion",
            "created": datetime.now()
        }
        
        converter = LiteLLMConverter()
        result = asyncio.run(converter.convert_response(mock_response))
        
        assert result.metadata["provider"] == "litellm"
        assert result.metadata["model"] == "gpt-4-turbo"
        assert result.metadata["finish_reason"] == "length"
        assert result.metadata["object"] == "chat.completion"

    @patch('pyagenity.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio 
    async def test_streaming_performance_simulation(self):
        """Test streaming performance with simulated chunks."""
        from pyagenity.adapters.llm.litellm_converter import LiteLLMConverter
        
        converter = LiteLLMConverter()
        
        # Create a large number of chunks to simulate high-throughput streaming
        chunks = []
        for i in range(100):
            chunk = Mock()
            chunk.id = f"chunk_{i}"
            chunk.choices = [Mock()]
            chunk.choices[0].delta = Mock()
            chunk.choices[0].delta.content = f"word{i} "
            chunk.choices[0].delta.reasoning_content = None
            chunk.choices[0].delta.tool_calls = None
            chunks.append(chunk)
        
        mock_stream = AsyncMock()
        mock_stream.__aiter__ = AsyncMock(return_value=iter(chunks))
        
        start_time = asyncio.get_event_loop().time()
        
        results = []
        async for item in converter._handle_stream(
            config={"thread_id": "perf_test"},
            node_name="perf_node", 
            stream=mock_stream
        ):
            results.append(item)
        
        end_time = asyncio.get_event_loop().time()
        
        # Should complete quickly even with many chunks
        assert end_time - start_time < 1.0  # Should take less than 1 second
        assert len(results) >= 1  # Should get at least final message