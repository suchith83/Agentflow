# Anthropic Converter Implementation Guide

**Date**: January 8, 2026  
**Status**: Implementation Specification  
**Priority**: HIGH

## Overview

This document provides the detailed implementation specification for adding Anthropic Claude support to agentflow using the official Anthropic Python SDK.

---

## Requirements

### Dependencies

```toml
# Add to pyproject.toml [project.optional-dependencies]
anthropic = ["anthropic>=0.40.0"]
```

### Supported Models

- `claude-3-5-sonnet-20241022` (Latest, best for most tasks)
- `claude-3-5-haiku-20241022` (Fast and efficient)
- `claude-3-opus-20240229` (Most capable, highest cost)
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

---

## File Structure

```
agentflow/adapters/llm/
├── __init__.py                    # Update: Add AnthropicConverter
├── base_converter.py              # Update: Add ANTHROPIC to ConverterType enum
├── model_response_converter.py   # Update: Add anthropic case
├── anthropic_converter.py         # NEW: Anthropic implementation
└── ...existing converters...

agentflow/graph/
├── agent.py                       # Update: Support Anthropic models
└── ...
```

---

## Implementation

### 1. Update ConverterType Enum

**File**: `agentflow/adapters/llm/base_converter.py`

```python
class ConverterType(str, Enum):
    """Enumeration of supported converter types for LLM responses."""

    OPENAI = "openai"
    LITELLM = "litellm"
    ANTHROPIC = "anthropic"  # ADD THIS
    GOOGLE = "google"
    CUSTOM = "custom"
```

### 2. Create Anthropic Converter

**File**: `agentflow/adapters/llm/anthropic_converter.py`

```python
"""
Converter for Anthropic Claude SDK responses to agentflow Message format.

This module provides conversion utilities for Anthropic's official Python SDK,
supporting both standard and streaming responses.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, cast

from agentflow.state.message import (
    Message,
    TokenUsages,
    generate_id,
)
from agentflow.state.message_block import (
    ImageBlock,
    MediaRef,
    TextBlock,
    ToolCallBlock,
)

from .base_converter import BaseConverter


logger = logging.getLogger("agentflow.adapters.anthropic")


try:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import MessageStreamEvent

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    AnthropicMessage = None  # type: ignore
    MessageStreamEvent = None  # type: ignore


class AnthropicConverter(BaseConverter):
    """
    Converter for Anthropic Claude responses to agentflow Message format.

    Handles both standard and streaming responses, extracting content,
    tool calls, and token usage details from Anthropic's Message objects.

    Supports:
    - Message responses (standard completions)
    - Streaming MessageStreamEvent responses
    - Image content (base64 and URLs)
    - Tool/function calls
    - Thinking/reasoning blocks
    """

    async def convert_response(self, response: AnthropicMessage) -> Message:  # type: ignore
        """
        Convert an Anthropic Message to an agentflow Message.

        Args:
            response: The Anthropic Message response object.

        Returns:
            Message: The converted message object.

        Raises:
            ImportError: If anthropic SDK is not installed.
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic SDK is not installed. "
                "Install it with: pip install 10xscale-agentflow[anthropic]"
            )

        # Extract usage information
        usage = response.usage
        usages = TokenUsages(
            completion_tokens=usage.output_tokens if usage else 0,
            prompt_tokens=usage.input_tokens if usage else 0,
            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
            cache_creation_input_tokens=getattr(
                usage, "cache_creation_input_tokens", 0
            ) if usage else 0,
            cache_read_input_tokens=getattr(
                usage, "cache_read_input_tokens", 0
            ) if usage else 0,
            reasoning_tokens=0,  # Anthropic doesn't separate reasoning tokens
        )

        # Extract content blocks
        blocks = []
        tools_calls = []
        reasoning_content = ""

        for content_block in response.content:
            # Handle text blocks
            if content_block.type == "text":
                text = content_block.text
                # Check if this is a thinking block (sometimes prefixed)
                if hasattr(content_block, "cache_control"):
                    # This is a thinking/reasoning block
                    reasoning_content = text
                else:
                    blocks.append(TextBlock(text=text))

            # Handle tool use blocks
            elif content_block.type == "tool_use":
                tool_call_id = content_block.id
                tool_name = content_block.name
                tool_args = content_block.input

                blocks.append(
                    ToolCallBlock(
                        name=tool_name,
                        args=tool_args,
                        id=tool_call_id,
                    )
                )

                tools_calls.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                )

            # Handle image blocks (if any)
            elif content_block.type == "image":
                # Anthropic returns images in responses for vision models
                if hasattr(content_block, "source"):
                    source = content_block.source
                    if source.type == "base64":
                        media = MediaRef(
                            kind="data",
                            data_base64=source.data,
                            mime_type=getattr(source, "media_type", "image/png"),
                        )
                        blocks.append(ImageBlock(media=media))
                    elif source.type == "url":
                        media = MediaRef(
                            kind="url",
                            url=source.url,
                        )
                        blocks.append(ImageBlock(media=media))

        logger.debug("Creating message from Anthropic response with id: %s", response.id)

        return Message(
            message_id=generate_id(response.id),
            role=response.role,  # Should be "assistant"
            content=blocks,
            reasoning=reasoning_content,
            timestamp=datetime.now().timestamp(),
            metadata={
                "provider": "anthropic",
                "model": response.model,
                "finish_reason": response.stop_reason or "UNKNOWN",
                "stop_sequence": response.stop_sequence,
            },
            usages=usages,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            tools_calls=tools_calls if tools_calls else None,
        )

    def _extract_delta_content_blocks(
        self,
        delta: Any,
    ) -> tuple[str, str, list]:
        """Extract content blocks from a streaming delta.

        Args:
            delta: Delta object from streaming response.

        Returns:
            tuple: (text_part, reasoning_part, content_blocks)
        """
        text_part = ""
        reasoning_part = ""
        content_blocks = []

        # Handle text deltas
        if hasattr(delta, "text") and delta.text:
            text_part = delta.text
            content_blocks.append(TextBlock(text=text_part))

        # Handle partial tool use
        if hasattr(delta, "partial_json") and delta.partial_json:
            # This is a partial tool call, we'll accumulate it
            pass

        return text_part, reasoning_part, content_blocks

    def _process_delta_tool_calls(
        self,
        event: Any,
        tool_calls: list,
        tool_ids: set,
        content_blocks: list,
        tool_call_buffer: dict,
    ) -> None:
        """Process tool calls from streaming event.

        Args:
            event: Event from streaming response.
            tool_calls: List to append tool calls to.
            tool_ids: Set to track tool call IDs.
            content_blocks: List to append tool call blocks to.
            tool_call_buffer: Buffer for accumulating partial tool calls.
        """
        # Handle tool_use start
        if event.type == "content_block_start":
            if hasattr(event, "content_block") and event.content_block.type == "tool_use":
                tool_block = event.content_block
                tool_call_buffer[tool_block.id] = {
                    "id": tool_block.id,
                    "name": tool_block.name,
                    "input": "",
                }

        # Handle tool_use delta (accumulate input)
        elif event.type == "content_block_delta":
            if hasattr(event.delta, "type") and event.delta.type == "input_json_delta":
                tool_id = tool_call_buffer.get("current_tool_id")
                if tool_id and tool_id in tool_call_buffer:
                    tool_call_buffer[tool_id]["input"] += event.delta.partial_json

        # Handle tool_use stop
        elif event.type == "content_block_stop":
            if event.index in tool_call_buffer:
                tool_data = tool_call_buffer[event.index]
                tool_id = tool_data["id"]

                if tool_id not in tool_ids:
                    tool_ids.add(tool_id)

                    try:
                        args = json.loads(tool_data["input"])
                    except json.JSONDecodeError:
                        args = {}

                    content_blocks.append(
                        ToolCallBlock(
                            name=tool_data["name"],
                            args=args,
                            id=tool_id,
                        )
                    )

                    tool_calls.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_data["name"],
                                "arguments": tool_data["input"],
                            },
                        }
                    )

    def _process_chunk(
        self,
        event: MessageStreamEvent | None,  # type: ignore
        seq: int,
        accumulated_content: str,
        accumulated_reasoning_content: str,
        tool_calls: list,
        tool_ids: set,
        tool_call_buffer: dict,
    ) -> tuple[str, str, list, int, Message | None]:
        """
        Process a single event from an Anthropic streaming response.

        Args:
            event: The current event from the stream.
            seq: Sequence number of the event.
            accumulated_content: Accumulated text content so far.
            accumulated_reasoning_content: Accumulated reasoning content so far.
            tool_calls: List of tool calls detected so far.
            tool_ids: Set of tool call IDs to avoid duplicates.
            tool_call_buffer: Buffer for accumulating partial tool calls.

        Returns:
            tuple: Updated accumulated content, reasoning, tool calls, sequence,
                and Message (if any).
        """
        if not event:
            return (
                accumulated_content,
                accumulated_reasoning_content,
                tool_calls,
                seq,
                None,
            )

        content_blocks = []

        # Handle content_block_delta events (text streaming)
        if event.type == "content_block_delta":
            if hasattr(event.delta, "type") and event.delta.type == "text_delta":
                text_part, reasoning_part, delta_blocks = self._extract_delta_content_blocks(
                    event.delta
                )
                accumulated_content += text_part
                accumulated_reasoning_content += reasoning_part
                content_blocks.extend(delta_blocks)

        # Handle tool use
        self._process_delta_tool_calls(
            event, tool_calls, tool_ids, content_blocks, tool_call_buffer
        )

        # Only create message if we have content
        if not content_blocks:
            return (
                accumulated_content,
                accumulated_reasoning_content,
                tool_calls,
                seq,
                None,
            )

        output_message = Message(
            message_id=generate_id(None),
            role="assistant",
            content=content_blocks,
            reasoning=accumulated_reasoning_content,
            tools_calls=tool_calls if tool_calls else None,
            delta=True,
        )

        return (
            accumulated_content,
            accumulated_reasoning_content,
            tool_calls,
            seq,
            output_message,
        )

    async def _handle_stream(
        self,
        config: dict,
        node_name: str,
        stream: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """
        Handle an Anthropic streaming response and yield Message objects for each event.

        Args:
            config: Node configuration parameters.
            node_name: Name of the node processing the response.
            stream: The Anthropic streaming response object.
            meta: Optional metadata for conversion.

        Yields:
            Message: Converted message chunk from the stream.
        """
        accumulated_content = ""
        tool_calls = []
        tool_ids = set()
        accumulated_reasoning_content = ""
        seq = 0
        tool_call_buffer = {}

        is_awaitable = inspect.isawaitable(stream)

        # Await stream if necessary
        if is_awaitable:
            stream = await stream

        # Anthropic SDK provides async iteration
        try:
            async for event in stream:  # type: ignore
                (
                    accumulated_content,
                    accumulated_reasoning_content,
                    tool_calls,
                    seq,
                    message,
                ) = self._process_chunk(
                    event,
                    seq,
                    accumulated_content,
                    accumulated_reasoning_content,
                    tool_calls,
                    tool_ids,
                    tool_call_buffer,
                )

                if message:
                    yield message
        except Exception as e:
            logger.error("Error during Anthropic streaming: %s", e)

        # After streaming, yield final message
        metadata = meta or {}
        metadata["provider"] = "anthropic"
        metadata["node_name"] = node_name
        metadata["thread_id"] = config.get("thread_id")

        blocks = []
        if accumulated_content:
            blocks.append(TextBlock(text=accumulated_content))
        if tool_calls:
            for tc in tool_calls:
                func_data = tc.get("function", {})
                try:
                    args = json.loads(func_data.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                blocks.append(
                    ToolCallBlock(
                        name=func_data.get("name", ""),
                        args=args,
                        id=tc.get("id", ""),
                    )
                )

        logger.debug(
            "Stream complete - Content: %s, Tool Calls: %s",
            accumulated_content,
            len(tool_calls),
        )

        message = Message(
            role="assistant",
            message_id=generate_id(None),
            content=blocks,
            delta=False,
            reasoning=accumulated_reasoning_content,
            tools_calls=tool_calls if tool_calls else None,
            metadata=metadata,
        )
        yield message

    async def convert_streaming_response(
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """
        Convert an Anthropic streaming or standard response to Message(s).

        Args:
            config: Node configuration parameters.
            node_name: Name of the node processing the response.
            response: The Anthropic response object (stream or standard).
            meta: Optional metadata for conversion.

        Yields:
            Message: Converted message(s) from the response.

        Raises:
            ImportError: If anthropic SDK is not installed.
            Exception: If response type is unsupported.
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "anthropic SDK is not installed. "
                "Install it with: pip install 10xscale-agentflow[anthropic]"
            )

        # Check if it's a standard Message response
        if HAS_ANTHROPIC and isinstance(response, AnthropicMessage):  # type: ignore
            message = await self.convert_response(cast(AnthropicMessage, response))  # type: ignore
            yield message
        # Otherwise assume it's a stream
        else:
            async for event in self._handle_stream(
                config or {},
                node_name or "",
                response,
                meta,
            ):
                yield event
```

### 3. Update ModelResponseConverter

**File**: `agentflow/adapters/llm/model_response_converter.py`

Add the anthropic case in `__init__`:

```python
def __init__(
    self,
    response: Any | Callable[..., Any],
    converter: BaseConverter | str,
) -> None:
    """
    Initialize ModelResponseConverter.

    Args:
        response (Any | Callable[..., Any]): The LLM response or a callable returning
            a response.
        converter (BaseConverter | str): Converter instance or string identifier
            (e.g., "litellm", "openai", "anthropic").

    Raises:
        ValueError: If the converter is not supported.
    """
    self.response = response

    if isinstance(converter, str) and converter == "litellm":
        from .litellm_converter import LiteLLMConverter

        self.converter = LiteLLMConverter()
        logger.debug("Using LiteLLMConverter for response conversion")

    elif isinstance(converter, str) and converter == "openai":
        from .openai_converter import OpenAIConverter

        self.converter = OpenAIConverter()
        logger.debug("Using OpenAIConverter for response conversion")
    
    # ADD THIS BLOCK
    elif isinstance(converter, str) and converter == "anthropic":
        from .anthropic_converter import AnthropicConverter

        self.converter = AnthropicConverter()
        logger.debug("Using AnthropicConverter for response conversion")
    
    elif isinstance(converter, BaseConverter):
        self.converter = converter
        logger.debug(f"Using custom converter: {type(converter).__name__}")
    else:
        logger.error(f"Unsupported converter: {converter}")
        raise ValueError(f"Unsupported converter: {converter}")
```

### 4. Update __init__.py

**File**: `agentflow/adapters/llm/__init__.py`

```python
"""
Integration adapters for optional third-party LLM SDKs.

This module exposes universal converter APIs to normalize responses and
streaming outputs from popular LLM SDKs (e.g., LiteLLM, OpenAI, Anthropic, Google GenAI)
for use in agentflow agent graphs.
"""

from .base_converter import BaseConverter, ConverterType
from .google_genai_converter import GoogleGenAIConverter
from .litellm_converter import LiteLLMConverter
from .anthropic_converter import AnthropicConverter  # ADD THIS


__all__ = [
    "BaseConverter",
    "ConverterType",
    "GoogleGenAIConverter",
    "LiteLLMConverter",
    "AnthropicConverter",  # ADD THIS
]
```

### 5. Update Agent Class

**File**: `agentflow/graph/agent.py`

Add automatic provider detection:

```python
def _detect_provider_from_model(model: str) -> str:
    """Detect the provider from the model string.
    
    Args:
        model: Model identifier string
        
    Returns:
        Provider name (openai, anthropic, google, etc.)
    """
    model_lower = model.lower()
    
    if model_lower.startswith("gpt-") or model_lower.startswith("o1-"):
        return "openai"
    elif model_lower.startswith("claude-"):
        return "anthropic"
    elif model_lower.startswith("gemini-"):
        return "google"
    else:
        # Default to openai for unknown models
        return "openai"


class Agent(BaseAgent):
    def __init__(
        self,
        model: str,
        system_prompt: list[dict[str, Any]],
        tools: list[Callable] | ToolNode | None = None,
        tool_node_name: str | None = None,
        extra_messages: list[Message] | None = None,
        trim_context: bool = False,
        tools_tags: set[str] | None = None,
        converter: str | None = None,  # ADD THIS PARAMETER
        **llm_kwargs,
    ):
        # ... existing initialization ...
        
        # Auto-detect converter if not provided
        if converter is None:
            converter = _detect_provider_from_model(model)
        
        self.converter = converter
        
        # Store for use in execute()
        self.llm_kwargs = llm_kwargs
        
    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
    ):
        # ... existing code ...
        
        # When returning the response, use detected converter
        return ModelResponseConverter(
            response,
            converter=self.converter,  # Use detected/specified converter
        )
```

### 6. Update pyproject.toml

**File**: `pyproject.toml`

```toml
[project.optional-dependencies]
litellm = ["litellm>=1.77.0"]
openai = ["openai>=1.50.0"]  # ADD THIS
anthropic = ["anthropic>=0.40.0"]  # ADD THIS
google-genai = ["google-genai>=1.56.0"]
# ... other dependencies ...
```

---

## Usage Examples

### Example 1: Basic Anthropic Agent

```python
from agentflow.graph import Agent, StateGraph
from agentflow.state import AgentState

# Create Anthropic agent - auto-detects from model name
agent = Agent(
    model="claude-3-5-sonnet-20241022",
    system_prompt="You are a helpful assistant",
)

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.set_entry_point("agent")

# Run
result = await graph.arun({"context": [{"role": "user", "content": "Hello!"}]})
```

### Example 2: Anthropic with Tools

```python
from agentflow.graph import Agent
from agentflow.prebuilt import tool

@tool
def calculator(operation: str, x: float, y: float) -> float:
    """Perform basic arithmetic operations."""
    if operation == "add":
        return x + y
    elif operation == "multiply":
        return x * y
    return 0

agent = Agent(
    model="claude-3-5-sonnet-20241022",
    system_prompt="You are a helpful calculator assistant",
    tools=[calculator],
)
```

### Example 3: Explicit Anthropic SDK Usage

```python
from anthropic import AsyncAnthropic
from agentflow.adapters.llm import AnthropicConverter

client = AsyncAnthropic(api_key="...")

async def custom_agent(state, config):
    response = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=state.context,
    )
    
    # Convert using agentflow converter
    converter = AnthropicConverter()
    message = await converter.convert_response(response)
    
    return {"context": [message]}
```

### Example 4: Streaming

```python
from anthropic import AsyncAnthropic
from agentflow.adapters.llm import AnthropicConverter

client = AsyncAnthropic(api_key="...")

async def streaming_agent(state, config):
    stream = await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=state.context,
        stream=True,
    )
    
    converter = AnthropicConverter()
    async for message in converter.convert_streaming_response(
        config=config,
        node_name="agent",
        response=stream,
    ):
        yield message
```

---

## Testing

### Unit Tests

**File**: `tests/adapters/test_anthropic_converter.py`

```python
import pytest
from agentflow.adapters.llm import AnthropicConverter
from agentflow.state.message import Message


@pytest.mark.asyncio
async def test_anthropic_text_response():
    """Test converting a simple text response."""
    converter = AnthropicConverter()
    
    # Mock Anthropic response
    mock_response = MockAnthropicMessage(
        id="msg_123",
        role="assistant",
        content=[MockTextBlock(text="Hello, world!")],
        model="claude-3-5-sonnet-20241022",
        usage=MockUsage(input_tokens=10, output_tokens=5),
    )
    
    message = await converter.convert_response(mock_response)
    
    assert isinstance(message, Message)
    assert message.role == "assistant"
    assert len(message.content) == 1
    assert message.content[0].text == "Hello, world!"
    assert message.usages.prompt_tokens == 10
    assert message.usages.completion_tokens == 5


@pytest.mark.asyncio
async def test_anthropic_tool_call():
    """Test converting a response with tool calls."""
    converter = AnthropicConverter()
    
    mock_response = MockAnthropicMessage(
        id="msg_456",
        role="assistant",
        content=[
            MockToolUseBlock(
                id="tool_123",
                name="calculator",
                input={"operation": "add", "x": 5, "y": 3},
            )
        ],
        model="claude-3-5-sonnet-20241022",
        usage=MockUsage(input_tokens=20, output_tokens=15),
    )
    
    message = await converter.convert_response(mock_response)
    
    assert len(message.content) == 1
    assert message.content[0].name == "calculator"
    assert message.content[0].args == {"operation": "add", "x": 5, "y": 3}
    assert message.tools_calls is not None
    assert len(message.tools_calls) == 1


# Add more tests for streaming, error handling, etc.
```

### Integration Tests

**File**: `tests/integration/test_anthropic_agent.py`

```python
import pytest
from agentflow.graph import Agent, StateGraph
from agentflow.state import AgentState


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anthropic_agent_basic():
    """Test basic Anthropic agent functionality."""
    agent = Agent(
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a helpful assistant",
    )
    
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.set_entry_point("agent")
    
    result = await graph.arun({
        "context": [{"role": "user", "content": "Say hello"}]
    })
    
    assert result is not None
    assert len(result["context"]) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anthropic_agent_with_tools():
    """Test Anthropic agent with tool calling."""
    from agentflow.prebuilt import tool
    
    @tool
    def get_weather(city: str) -> str:
        return f"Weather in {city}: Sunny, 72°F"
    
    agent = Agent(
        model="claude-3-5-sonnet-20241022",
        system_prompt="You are a weather assistant",
        tools=[get_weather],
    )
    
    # ... test implementation ...
```

---

## Documentation

### Tutorial Addition

**File**: `docs/Tutorial/anthropic-integration.md`

```markdown
# Using Anthropic Claude Models

AgentFlow supports Anthropic's Claude models through the official Anthropic Python SDK.

## Installation

```bash
pip install 10xscale-agentflow[anthropic]
```

## Basic Usage

```python
from agentflow.graph import Agent

agent = Agent(
    model="claude-3-5-sonnet-20241022",
    system_prompt="You are a helpful assistant",
)
```

## Supported Models

- **Claude 3.5 Sonnet**: Best balance of intelligence and speed
- **Claude 3.5 Haiku**: Fastest and most cost-effective
- **Claude 3 Opus**: Most capable for complex tasks

## Advanced Features

### Tool Calling

Claude supports function/tool calling natively...

### Streaming

Anthropic responses can be streamed...
```

---

## Checklist

- [ ] Create `anthropic_converter.py`
- [ ] Update `base_converter.py` (add ANTHROPIC enum)
- [ ] Update `model_response_converter.py` (add anthropic case)
- [ ] Update `__init__.py` (export AnthropicConverter)
- [ ] Update `agent.py` (auto-detect provider)
- [ ] Update `pyproject.toml` (add anthropic dependency)
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update documentation
- [ ] Add examples
- [ ] Test with real Anthropic API

---

## Timeline

- **Day 1**: Create converter, update files
- **Day 2**: Write tests, fix bugs
- **Day 3**: Documentation and examples
- **Day 4**: Integration testing and polish

---

**Questions? Contact the AgentFlow team.**
