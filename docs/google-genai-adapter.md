# Google Generative AI Adapter for Agentflow

The Google Generative AI adapter provides seamless integration between Google's `google-genai` SDK and the Agentflow framework.

## Overview

The `GoogleGenAIConverter` class converts Google GenAI responses (both standard and streaming) into Agentflow's internal `Message` format, enabling you to use Google's Gemini models within your multi-agent workflows.

## Features

- ✅ **Standard Response Conversion**: Convert `GenerateContentResponse` objects to Agentflow Messages
- ✅ **Streaming Support**: Handle streaming responses with real-time message chunks
- ✅ **Function Calling**: Full support for Google GenAI function/tool calling
- ✅ **Multimodal Content**: Handle text, images, audio, and video content
- ✅ **Reasoning/Thoughts**: Extract and preserve reasoning/thought content
- ✅ **Token Usage Tracking**: Comprehensive token usage and cost tracking
- ✅ **Metadata Preservation**: Preserve model version, finish reasons, and other metadata

## Installation

Install the Google GenAI SDK:

```bash
pip install google-genai
```

Or install with Agentflow's optional dependencies:

```bash
pip install agentflow[google-genai]
```

## Quick Start

### Setup

1. Get your API key from [Google AI Studio](https://ai.google.dev/)
2. Set your environment variable:

```bash
export GEMINI_API_KEY='your-api-key-here'
# or
export GOOGLE_API_KEY='your-api-key-here'
```

### Basic Usage

```python
import asyncio
from google import genai
from google.genai import types
from agentflow.adapters.llm import GoogleGenAIConverter

async def main():
    # Create Google GenAI client
    client = genai.Client(api_key='your-api-key')
    
    # Generate content
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents='Write a haiku about coding',
    )
    
    # Convert to Agentflow Message
    converter = GoogleGenAIConverter()
    message = await converter.convert_response(response)
    
    print(f"Message: {message.content[0].text}")
    print(f"Tokens used: {message.usages.total_tokens}")
    
    client.close()

asyncio.run(main())
```

## Usage Examples

### Streaming Responses

```python
async def streaming_example():
    client = genai.Client(api_key='your-api-key')
    
    # Generate streaming content
    stream = client.models.generate_content_stream(
        model='gemini-2.0-flash-exp',
        contents='Tell me a story',
    )
    
    # Convert stream
    converter = GoogleGenAIConverter()
    config = {'thread_id': 'my-thread'}
    
    async for message in converter.convert_streaming_response(
        config=config,
        node_name='story_node',
        response=stream,
    ):
        if message.delta:
            # Streaming chunk
            for block in message.content:
                if hasattr(block, 'text'):
                    print(block.text, end='', flush=True)
        else:
            # Final message
            print(f"\n\nTotal tokens: {message.usages.total_tokens}")
    
    client.close()
```

### Function Calling

```python
async def function_calling_example():
    client = genai.Client(api_key='your-api-key')
    
    # Define a function
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: sunny, 72°F"
    
    # Generate with function calling
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents="What's the weather in Boston?",
        config=types.GenerateContentConfig(
            tools=[get_weather],
        ),
    )
    
    # Convert response
    converter = GoogleGenAIConverter()
    message = await converter.convert_response(response)
    
    # Check for tool calls
    if message.tools_calls:
        for tool_call in message.tools_calls:
            print(f"Function: {tool_call['function']['name']}")
            print(f"Arguments: {tool_call['function']['arguments']}")
    
    client.close()
```

### Multimodal Content

```python
async def multimodal_example():
    from google.genai import types
    
    client = genai.Client(api_key='your-api-key')
    
    # Generate content with image
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=[
            'Describe this image',
            types.Part.from_uri(
                file_uri='gs://bucket/image.jpg',
                mime_type='image/jpeg',
            ),
        ],
    )
    
    # Convert response
    converter = GoogleGenAIConverter()
    message = await converter.convert_response(response)
    
    # Access content blocks
    for block in message.content:
        if hasattr(block, 'text'):
            print(f"Text: {block.text}")
        elif hasattr(block, 'media'):
            print(f"Media: {block.media.url or 'inline'}")
    
    client.close()
```

### Using in Agentflow Nodes

```python
from agentflow.graph import StateGraph
from agentflow.state import AgentState
from google import genai
from google.genai import types
from agentflow.adapters.llm import GoogleGenAIConverter

# Initialize client globally
google_client = genai.Client(api_key='your-api-key')
converter = GoogleGenAIConverter()

async def google_genai_node(state: AgentState, config: dict) -> list:
    """Node that uses Google GenAI."""
    # Get user message
    user_message = state.context[-1].content[0].text
    
    # Call Google GenAI
    response = google_client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=user_message,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1000,
        ),
    )
    
    # Convert to Message
    message = await converter.convert_response(response)
    
    return [message]

# Build graph
graph = StateGraph()
graph.add_node("google_node", google_genai_node)
graph.set_entry_point("google_node")
graph.add_edge("google_node", END)

# Use the graph
compiled = graph.compile()
result = await compiled.ainvoke({
    "context": [Message.text_message("Hello!")]
})
```

## Supported Features

### Content Types

The converter handles all Google GenAI content types:

- **Text**: Plain text responses
- **Thoughts**: Reasoning/thinking content
- **Function Calls**: Tool/function invocations
- **Images**: Inline data or file URIs
- **Audio**: Inline data or file URIs
- **Video**: Inline data or file URIs

### Message Structure

Converted messages include:

- `message_id`: Unique identifier
- `role`: Always "assistant" for Google GenAI responses
- `content`: List of content blocks (TextBlock, ToolCallBlock, etc.)
- `reasoning`: Extracted reasoning/thought content
- `timestamp`: Response creation time
- `metadata`: Provider info, model version, finish reason
- `usages`: Token usage statistics
- `tools_calls`: Function/tool call information

### Token Usage

The converter extracts token usage information:

```python
message.usages.completion_tokens  # Output tokens
message.usages.prompt_tokens      # Input tokens
message.usages.total_tokens       # Total tokens
message.usages.cache_read_input_tokens  # Cached tokens
```

## Configuration Options

### Converter Initialization

```python
# Basic initialization
converter = GoogleGenAIConverter()

# With state (for context-aware conversion)
converter = GoogleGenAIConverter(state=agent_state)
```

### Streaming Configuration

```python
config = {
    'thread_id': 'unique-thread-id',  # Thread identifier
    # Add custom config as needed
}

async for message in converter.convert_streaming_response(
    config=config,
    node_name='my_node',
    response=stream,
    meta={'custom': 'metadata'},  # Optional metadata
):
    process_message(message)
```

## Error Handling

The converter handles common errors gracefully:

```python
try:
    message = await converter.convert_response(response)
except ImportError as e:
    # google-genai not installed
    print(f"Missing dependency: {e}")
except Exception as e:
    # Other errors
    print(f"Conversion error: {e}")
```

## Best Practices

1. **API Key Management**: Always use environment variables for API keys
2. **Client Lifecycle**: Create a single client instance and reuse it
3. **Close Clients**: Always close clients after use or use context managers
4. **Error Handling**: Handle ImportError and conversion errors appropriately
5. **Streaming**: Use streaming for long responses to improve user experience
6. **Token Tracking**: Monitor token usage to manage costs

## Advanced Features

### Custom Metadata

Pass custom metadata through the conversion:

```python
meta = {
    'user_id': '123',
    'session_id': 'abc',
    'custom_field': 'value',
}

message = await converter.convert_response(response)
# Or for streaming:
async for message in converter.convert_streaming_response(
    config=config,
    node_name='node',
    response=stream,
    meta=meta,
):
    print(message.metadata)
```

### Context-Aware Conversion

Initialize the converter with agent state for context-aware processing:

```python
converter = GoogleGenAIConverter(state=agent_state)
message = await converter.convert_response(response)
```

## API Reference

### GoogleGenAIConverter

```python
class GoogleGenAIConverter(BaseConverter):
    """Converter for Google GenAI responses."""
    
    def __init__(self, state: AgentState | None = None) -> None:
        """Initialize converter with optional state."""
    
    async def convert_response(
        self, 
        response: GenerateContentResponse
    ) -> Message:
        """Convert standard response to Message."""
    
    async def convert_streaming_response(
        self,
        config: dict,
        node_name: str,
        response: Any,
        meta: dict | None = None,
    ) -> AsyncGenerator[Message]:
        """Convert streaming response to Messages."""
```

## Examples

See the complete example in `examples/google_genai_example.py`:

```bash
# Set your API key
export GEMINI_API_KEY='your-api-key'

# Run the examples
python examples/google_genai_example.py
```

## Troubleshooting

### ImportError: google-genai not installed

**Solution**: Install the package:
```bash
pip install google-genai
```

### API Key Error

**Solution**: Ensure your API key is set:
```bash
export GEMINI_API_KEY='your-api-key'
```

### Empty Responses

**Issue**: Response has no candidates
**Solution**: Check your prompt and model parameters

### Streaming Issues

**Issue**: Stream not working
**Solution**: Ensure you're using `generate_content_stream` not `generate_content`

## Resources

- [Google GenAI SDK Documentation](https://googleapis.github.io/python-genai/)
- [Google AI Studio](https://ai.google.dev/)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Agentflow Documentation](https://github.com/10xHub/Agentflow)

## License

This adapter is part of the Agentflow framework and follows the same license.
