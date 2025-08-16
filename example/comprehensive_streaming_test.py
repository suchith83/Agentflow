"""
Comprehensive streaming test for PyAgenity.
Tests both streamable and non-streamable responses.
"""

import asyncio

from pyagenity.graph.graph import StateGraph
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import Message


def string_agent(state, config, checkpointer=None, store=None):
    """Agent that returns a simple string response."""
    return "Hello from string agent! This response will be streamed character by character."


def dict_agent(state, config, checkpointer=None, store=None):
    """Agent that returns a dict response."""
    return {"role": "assistant", "content": "Hello from dict agent! This will also be streamed."}


def agentstate_agent(state, config, checkpointer=None, store=None):
    """Agent that returns an AgentState (non-streamable)."""
    new_state = AgentState()
    new_state.context.append(Message.from_text("This came from AgentState agent"))
    return new_state


def none_agent(state, config, checkpointer=None, store=None):
    """Agent that returns None."""
    return


async def test_string_streaming():
    """Test streaming with string response."""
    print("\\n=== Testing String Response Streaming ===")

    graph = StateGraph()
    graph.add_node("STRING_NODE", string_agent)
    graph.set_entry_point("STRING_NODE")
    app = graph.compile()

    inp = {"messages": [Message.from_text("Test string streaming")]}
    config = {"thread_id": "string_test", "recursion_limit": 3}

    chunks = []
    async for chunk in app.astream(inp, config=config):
        chunks.append(chunk)
        if len(chunk.content) < 100:  # Only print shorter content
            print(f"  CHUNK: '{chunk.content}' (final: {chunk.is_final})")
        else:
            print(f"  CHUNK: '{chunk.content[:50]}...' (final: {chunk.is_final})")

    print(f"  Total chunks: {len(chunks)}")
    final_content = chunks[-2].content if len(chunks) > 1 else chunks[-1].content
    print(f"  Final content: '{final_content}'")


async def test_dict_streaming():
    """Test streaming with dict response."""
    print("\\n=== Testing Dict Response Streaming ===")

    graph = StateGraph()
    graph.add_node("DICT_NODE", dict_agent)
    graph.set_entry_point("DICT_NODE")
    app = graph.compile()

    inp = {"messages": [Message.from_text("Test dict streaming")]}
    config = {"thread_id": "dict_test", "recursion_limit": 3}

    chunks = []
    async for chunk in app.astream(inp, config=config):
        chunks.append(chunk)
        if len(chunk.content) < 100:
            print(f"  CHUNK: '{chunk.content}' (final: {chunk.is_final})")
        else:
            print(f"  CHUNK: '{chunk.content[:50]}...' (final: {chunk.is_final})")

    print(f"  Total chunks: {len(chunks)}")
    final_content = chunks[-2].content if len(chunks) > 1 else chunks[-1].content
    print(f"  Final content: '{final_content}'")


async def test_agentstate_streaming():
    """Test streaming with AgentState response (non-streamable)."""
    print("\\n=== Testing AgentState Response Streaming ===")

    graph = StateGraph()
    graph.add_node("STATE_NODE", agentstate_agent)
    graph.set_entry_point("STATE_NODE")
    app = graph.compile()

    inp = {"messages": [Message.from_text("Test AgentState streaming")]}
    config = {"thread_id": "state_test", "recursion_limit": 3}

    chunks = []
    async for chunk in app.astream(inp, config=config):
        chunks.append(chunk)
        print(f"  CHUNK: '{chunk.content[:50]}...' (final: {chunk.is_final})")

    print(f"  Total chunks: {len(chunks)}")


async def test_none_streaming():
    """Test streaming with None response."""
    print("\\n=== Testing None Response Streaming ===")

    graph = StateGraph()
    graph.add_node("NONE_NODE", none_agent)
    graph.set_entry_point("NONE_NODE")
    app = graph.compile()

    inp = {"messages": [Message.from_text("Test None streaming")]}
    config = {"thread_id": "none_test", "recursion_limit": 3}

    chunks = []
    async for chunk in app.astream(inp, config=config):
        chunks.append(chunk)
        print(f"  CHUNK: '{chunk.content}' (final: {chunk.is_final})")

    print(f"  Total chunks: {len(chunks)}")


async def test_multi_node_streaming():
    """Test streaming with multiple nodes."""
    print("\\n=== Testing Multi-Node Streaming ===")

    graph = StateGraph()
    graph.add_node("FIRST", string_agent)
    graph.add_node("SECOND", dict_agent)
    graph.set_entry_point("FIRST")
    graph.add_edge("FIRST", "SECOND")
    app = graph.compile()

    inp = {"messages": [Message.from_text("Test multi-node streaming")]}
    config = {"thread_id": "multi_test", "recursion_limit": 5}

    chunks = []
    current_node = "start"
    async for chunk in app.astream(inp, config=config):
        chunks.append(chunk)
        print(f"  CHUNK: '{chunk.content[:50]}...' (final: {chunk.is_final})")

        # Stop after reasonable number of chunks
        if len(chunks) > 20:
            break

    print(f"  Total chunks: {len(chunks)}")


async def main():
    """Run all streaming tests."""
    print("PyAgenity Streaming Tests")
    print("=" * 50)

    try:
        await test_string_streaming()
        await test_dict_streaming()
        await test_agentstate_streaming()
        await test_none_streaming()
        await test_multi_node_streaming()

        print("\\n" + "=" * 50)
        print("✅ All streaming tests completed successfully!")

    except Exception as e:
        print(f"\\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
