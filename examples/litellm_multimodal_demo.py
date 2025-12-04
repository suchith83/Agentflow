"""Demo script showing the new multimodal support in LiteLLM converter.

This demonstrates that the converter now properly handles:
1. Audio content (data + transcript)
2. Image content (URL references)
3. Reasoning content (thinking/reasoning blocks)
"""

import asyncio
from agentflow.adapters.llm.litellm_converter import LiteLLMConverter


class MockModelResponse:
    """Mock ModelResponse for demo."""

    def __init__(self, data):
        self.id = data.get("id", "demo_id")
        self._data = data

    def model_dump(self):
        return self._data


async def demo_audio_support():
    """Demo audio content extraction."""
    print("\n=== Audio Support Demo ===")

    converter = LiteLLMConverter()

    response_data = {
        "id": "audio_demo",
        "choices": [
            {
                "message": {
                    "content": "Here's an audio response",
                    "audio": {
                        "id": "audio_123",
                        "data": "SGVsbG8gV29ybGQh",  # Base64 "Hello World!"
                        "transcript": "Hello World!",
                        "expires_at": 1234567890,
                    },
                }
            }
        ],
        "usage": {},
    }

    response = MockModelResponse(response_data)
    message = await converter.convert_response(response)

    print(f"Message ID: {message.message_id}")
    print(f"Number of content blocks: {len(message.content)}")
    for i, block in enumerate(message.content):
        print(f"  Block {i}: {block.type}")
        if block.type == "text":
            print(f"    Text: {block.text}")
        elif block.type == "audio":
            print(f"    Transcript: {block.transcript}")
            print(f"    Has audio data: {bool(block.media.data_base64)}")


async def demo_image_support():
    """Demo image content extraction."""
    print("\n=== Image Support Demo ===")

    converter = LiteLLMConverter()

    response_data = {
        "id": "image_demo",
        "choices": [
            {
                "message": {
                    "content": "Check out these images",
                    "images": [
                        {"url": "https://example.com/cat.jpg"},
                        {"url": "https://example.com/dog.jpg"},
                    ],
                }
            }
        ],
        "usage": {},
    }

    response = MockModelResponse(response_data)
    message = await converter.convert_response(response)

    print(f"Message ID: {message.message_id}")
    print(f"Number of content blocks: {len(message.content)}")
    for i, block in enumerate(message.content):
        print(f"  Block {i}: {block.type}")
        if block.type == "text":
            print(f"    Text: {block.text}")
        elif block.type == "image":
            print(f"    Image URL: {block.media.url}")


async def demo_reasoning_support():
    """Demo reasoning content extraction."""
    print("\n=== Reasoning Support Demo ===")

    converter = LiteLLMConverter()

    response_data = {
        "id": "reasoning_demo",
        "choices": [
            {
                "message": {
                    "content": "The answer is 42",
                    "reasoning_content": "I calculated this by analyzing the question deeply",
                }
            }
        ],
        "usage": {},
    }

    response = MockModelResponse(response_data)
    message = await converter.convert_response(response)

    print(f"Message ID: {message.message_id}")
    print(f"Number of content blocks: {len(message.content)}")
    print(f"Reasoning field: {message.reasoning}")
    for i, block in enumerate(message.content):
        print(f"  Block {i}: {block.type}")
        if block.type == "text":
            print(f"    Text: {block.text}")
        elif block.type == "reasoning":
            print(f"    Reasoning: {block.summary}")


async def demo_multimodal_combined():
    """Demo all content types combined."""
    print("\n=== Combined Multimodal Demo ===")

    converter = LiteLLMConverter()

    response_data = {
        "id": "multimodal_demo",
        "choices": [
            {
                "message": {
                    "content": "Let me explain with audio and images",
                    "reasoning_content": "I thought about the best way to present this",
                    "audio": {
                        "id": "audio_456",
                        "data": "RXhwbGFuYXRpb24=",  # Base64
                        "transcript": "This is my explanation",
                    },
                    "images": [
                        {"url": "https://example.com/diagram.png"},
                    ],
                }
            }
        ],
        "usage": {},
    }

    response = MockModelResponse(response_data)
    message = await converter.convert_response(response)

    print(f"Message ID: {message.message_id}")
    print(f"Number of content blocks: {len(message.content)}")
    print(f"Reasoning field: {message.reasoning}")
    for i, block in enumerate(message.content):
        print(f"  Block {i}: {block.type}")
        if block.type == "text":
            print(f"    Text: {block.text}")
        elif block.type == "reasoning":
            print(f"    Summary: {block.summary}")
        elif block.type == "audio":
            print(f"    Transcript: {block.transcript}")
        elif block.type == "image":
            print(f"    Image URL: {block.media.url}")


async def main():
    """Run all demos."""
    print("=" * 60)
    print("LiteLLM Converter Multimodal Support Demo")
    print("=" * 60)

    await demo_audio_support()
    await demo_image_support()
    await demo_reasoning_support()
    await demo_multimodal_combined()

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
