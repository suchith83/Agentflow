"""
Simple test for streaming functionality.
"""

import asyncio

from pyagenity.graph.graph import StateGraph
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import END, Message


def simple_agent(state, config, checkpointer=None, store=None):
    """A simple agent that returns a text response."""
    return "This is a simple response that will be chunked for streaming simulation."


def test_basic_streaming():
    """Test basic streaming functionality with non-streaming response."""
    print("Testing basic streaming...")

    # Create a simple graph
    graph = StateGraph()
    graph.add_node("MAIN", simple_agent)
    graph.set_entry_point("MAIN")

    # Compile the graph
    app = graph.compile()

    # Test input
    inp = {"messages": [Message.from_text("Hello")]}
    config = {"thread_id": "test", "recursion_limit": 5}

    print("\\nStreaming output:")
    try:
        for chunk in app.stream(inp, config=config):
            if chunk.delta:
                print(f"[{chunk.delta}]", end=" ", flush=True)
            if chunk.is_final:
                print("\\n✓ Stream completed successfully")
                break
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


async def test_basic_async_streaming():
    """Test basic async streaming functionality."""
    print("\\nTesting async streaming...")

    # Create a simple graph
    graph = StateGraph()
    graph.add_node("MAIN", simple_agent)
    graph.set_entry_point("MAIN")

    # Compile the graph
    app = graph.compile()

    # Test input
    inp = {"messages": [Message.from_text("Hello async")]}
    config = {"thread_id": "async_test", "recursion_limit": 5}

    print("\\nAsync streaming output:")
    try:
        async for chunk in app.astream(inp, config=config):
            if chunk.delta:
                print(f"[{chunk.delta}]", end=" ", flush=True)
            if chunk.is_final:
                print("\\n✓ Async stream completed successfully")
                break
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_basic_streaming()
    asyncio.run(test_basic_async_streaming())
