"""
Debug test for streaming functionality.
"""

import asyncio

from pyagenity.graph.graph import StateGraph
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import Message
from pyagenity.graph.utils.streaming import extract_content_from_response


def debug_agent(state, config, checkpointer=None, store=None):
    """A simple agent that returns a text response."""
    response = "This is a simple response that will be chunked for streaming simulation."
    print(f"DEBUG: Agent returning: {response}")
    return response


def test_content_extraction():
    """Test content extraction from different response types."""
    print("Testing content extraction...")

    # Test with string
    result1 = "Hello world"
    content1 = extract_content_from_response(result1)
    print(f"String result: '{result1}' -> content: '{content1}'")

    # Test with dict
    result2 = {"content": "Hello from dict"}
    content2 = extract_content_from_response(result2)
    print(f"Dict result: {result2} -> content: '{content2}'")

    # Test with something else
    result3 = AgentState()
    content3 = extract_content_from_response(result3)
    print(f"AgentState result: {type(result3)} -> content: '{content3}'")


def test_debug_streaming():
    """Test streaming with debug output."""
    print("\\nTesting debug streaming...")

    # Create a simple graph
    graph = StateGraph()
    graph.add_node("MAIN", debug_agent)
    graph.set_entry_point("MAIN")

    # Compile the graph
    app = graph.compile()

    # Test input
    inp = {"messages": [Message.from_text("Hello")]}
    config = {"thread_id": "debug_test", "recursion_limit": 5}

    print("\\nDebug streaming output:")
    chunk_count = 0
    try:
        for chunk in app.stream(inp, config=config):
            chunk_count += 1
            print(
                f"CHUNK {chunk_count}: content='{chunk.content[:50]}...' delta='{chunk.delta[:50]}...' final={chunk.is_final}"
            )
            if chunk.is_final or chunk_count > 10:  # Limit output for debugging
                print("✓ Stream completed successfully")
                break
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_content_extraction()
    test_debug_streaming()
