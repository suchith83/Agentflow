"""
Simple test to debug the streaming issue.
"""

import asyncio

from pyagenity.graph.graph import StateGraph
from pyagenity.graph.utils import Message


def debug_agent(state, config, checkpointer=None, store=None):
    """A simple agent that returns a text response."""
    response = "This is a simple response from MAIN node."
    print(f"DEBUG: MAIN Agent returning: {response}")
    return response


async def test_simple():
    """Simple test to debug streaming execution."""
    # Create a simple graph
    graph = StateGraph()
    graph.add_node("MAIN", debug_agent)
    graph.set_entry_point("MAIN")

    # Compile the graph
    app = graph.compile()

    # Test input
    inp = {"messages": [Message.from_text("Hello")]}
    config = {"thread_id": "debug_test", "recursion_limit": 5}

    print("\\nSimple streaming test:")
    chunk_count = 0
    try:
        async for chunk in app.astream(inp, config=config):
            chunk_count += 1
            print(f"CHUNK {chunk_count}: '{chunk.content[:50]}...' final={chunk.is_final}")
            if chunk_count > 15:  # Only limit by count, not by is_final
                print("✓ Stream completed successfully")
                break
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_simple())
