"""Multi-Modal Agent Example

This example demonstrates how to use the Agent class with different output types
to create multi-agent workflows that generate text, images, videos, and audio.

The explicit output_type parameter makes it clear what each agent produces,
enabling complex multi-modal workflows following the Google ADK pattern.
"""

import asyncio
import logging

from agentflow.graph import Agent, StateGraph
from agentflow.state import AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalState(AgentState):
    """State for multi-modal agent workflow."""

    prompt: str = ""
    enhanced_prompt: str = ""
    image_url: str = ""
    video_url: str = ""


async def example_1_text_to_image_workflow():
    """Example 1: Text agent generates prompt -> Image agent generates image."""
    print("\n" + "=" * 80)
    print("Example 1: Text-to-Image Workflow (Google ADK Style)")
    print("=" * 80)

    # Agent 1: Prompt engineering (text generation)
    prompt_agent = Agent(
        model="gemini-2.0-flash-exp",
        provider="google",
        output_type="text",  # Default, but explicit for clarity
        system_prompt=[
            {
                "role": "system",
                "content": "You are a prompt engineering expert. "
                "Generate detailed, creative image prompts.",
            }
        ],
    )

    # Agent 2: Image generation
    image_agent = Agent(
        model="imagen-3.0-generate-001",
        provider="google",
        output_type="image",  # Explicit: this agent generates images
    )

    # Build the graph
    graph = StateGraph[MultiModalState](MultiModalState())

    graph.add_node("PROMPT_ENGINEER", prompt_agent)
    graph.add_node("IMAGE_GENERATOR", image_agent)

    graph.add_edge("PROMPT_ENGINEER", "IMAGE_GENERATOR")
    graph.set_entry_point("PROMPT_ENGINEER")

    app = graph.compile()

    print("\nWorkflow: User Input -> Prompt Engineer -> Image Generator")
    print("✓ Configured successfully!")


async def example_2_openai_multimodal():
    """Example 2: OpenAI multi-modal agents (text, image, audio)."""
    print("\n" + "=" * 80)
    print("Example 2: OpenAI Multi-Modal Agents")
    print("=" * 80)

    # Text agent (default)
    text_agent = Agent(
        model="gpt-4o",
        provider="openai",
        # output_type="text" is default, no need to specify
        system_prompt=[{"role": "system", "content": "You are a helpful assistant."}],
    )

    # Image agent (DALL-E)
    image_agent = Agent(
        model="dall-e-3",
        provider="openai",
        output_type="image",  # Explicit: generates images
    )

    # Audio agent (TTS)
    audio_agent = Agent(
        model="tts-1",
        provider="openai",
        output_type="audio",  # Explicit: generates audio
    )

    print("\nConfigured 3 OpenAI agents:")
    print("  1. Text Agent (gpt-4o) - output_type='text'")
    print("  2. Image Agent (dall-e-3) - output_type='image'")
    print("  3. Audio Agent (tts-1) - output_type='audio'")
    print("✓ All agents ready!")


async def example_3_google_full_multimodal():
    """Example 3: Full Google multi-modal pipeline (text -> image -> video -> audio)."""
    print("\n" + "=" * 80)
    print("Example 3: Google Full Multi-Modal Pipeline")
    print("=" * 80)

    # Agent 1: Text generation
    text_agent = Agent(
        model="gemini-2.0-flash-exp",
        provider="google",
        output_type="text",
        system_prompt=[
            {"role": "system", "content": "Generate creative content descriptions."}
        ],
    )

    # Agent 2: Image generation
    image_agent = Agent(
        model="imagen-3.0-generate-001",
        provider="google",
        output_type="image",
    )

    # Agent 3: Video generation
    video_agent = Agent(
        model="veo-2.0",
        provider="google",
        output_type="video",
    )

    # Agent 4: Audio generation (TTS)
    audio_agent = Agent(
        model="gemini-2.5-flash-preview-tts",
        provider="google",
        output_type="audio",
    )

    # Build the graph
    graph = StateGraph[MultiModalState](MultiModalState())

    graph.add_node("TEXT", text_agent)
    graph.add_node("IMAGE", image_agent)
    graph.add_node("VIDEO", video_agent)
    graph.add_node("AUDIO", audio_agent)

    # Sequential pipeline
    graph.add_edge("TEXT", "IMAGE")
    graph.add_edge("IMAGE", "VIDEO")
    graph.add_edge("VIDEO", "AUDIO")
    graph.set_entry_point("TEXT")

    app = graph.compile()

    print("\nPipeline: TEXT -> IMAGE -> VIDEO -> AUDIO")
    print("  1. Gemini generates description")
    print("  2. Imagen generates image")
    print("  3. Veo generates video")
    print("  4. Gemini TTS generates narration")
    print("✓ Full multi-modal pipeline configured!")


async def example_4_third_party_models():
    """Example 4: Third-party models (Qwen, DeepSeek, Ollama) - all text by default."""
    print("\n" + "=" * 80)
    print("Example 4: Third-Party Models (Text Generation)")
    print("=" * 80)

    # Qwen (via OpenAI-compatible API)
    qwen_agent = Agent(
        model="qwen-2.5-72b-instruct",
        provider="openai",
        base_url="https://api.qwen.com/v1",
        # output_type="text" is default - no need to specify
    )

    # DeepSeek
    deepseek_agent = Agent(
        model="deepseek-chat",
        provider="openai",
        base_url="https://api.deepseek.com/v1",
    )

    # Ollama (local)
    ollama_agent = Agent(
        model="llama3:70b",
        provider="openai",
        base_url="http://localhost:11434/v1",
    )

    print("\nConfigured 3 third-party agents (all text generation):")
    print("  1. Qwen 2.5 72B - output_type='text' (default)")
    print("  2. DeepSeek Chat - output_type='text' (default)")
    print("  3. Ollama Llama3 - output_type='text' (default)")
    print("✓ All agents use default text output type!")


async def example_5_latest_google_models():
    """Example 5: Latest Google models with any naming convention."""
    print("\n" + "=" * 80)
    print("Example 5: Latest Google Models (No Name Parsing Required!)")
    print("=" * 80)

    # Latest models - just specify output_type explicitly
    agents = [
        Agent(
            model="models/gemini-3-pro-image-preview",
            provider="google",
            output_type="text",  # We tell it what we want
        ),
        Agent(
            model="models/gemini-2.5-flash-image",
            provider="google",
            output_type="image",  # Explicit
        ),
        Agent(
            model="models/gemini-2.5-flash-native-audio-preview-12-2025",
            provider="google",
            output_type="audio",  # Explicit
        ),
        Agent(
            model="models/gemini-2.5-flash-preview-tts",
            provider="google",
            output_type="audio",  # Explicit
        ),
    ]

    print("\nConfigured 4 latest Google models:")
    print("  1. gemini-3-pro-image-preview - output_type='text'")
    print("  2. gemini-2.5-flash-image - output_type='image'")
    print("  3. gemini-2.5-flash-native-audio (realtime) - output_type='audio'")
    print("  4. gemini-2.5-flash-preview-tts - output_type='audio'")
    print("\n✓ No model name parsing needed!")
    print("✓ Works with ANY model name - just specify output_type!")


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("MULTI-MODAL AGENT EXAMPLES")
    print("Demonstrating the explicit output_type parameter")
    print("=" * 80)

    try:
        # Run all examples
        await example_1_text_to_image_workflow()
        await example_2_openai_multimodal()
        await example_3_google_full_multimodal()
        await example_4_third_party_models()
        await example_5_latest_google_models()

        print("\n" + "=" * 80)
        print("KEY TAKEAWAYS")
        print("=" * 80)
        print("\n1. ✅ Explicit output_type parameter (text, image, video, audio)")
        print("2. ✅ No model name parsing - works with ANY model")
        print("3. ✅ Default is 'text' - most common case")
        print("4. ✅ Multi-agent workflows like Google ADK")
        print("5. ✅ Future-proof - new models work automatically")
        print("\n" + "=" * 80)
        print("✓ All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

