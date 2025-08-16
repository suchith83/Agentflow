#!/usr/bin/env python3
"""
Final demonstration of PyAgenity streaming functionality.
This demo shows stream and astream methods working with various response types.
"""

import asyncio
from pyagenity.graph.graph.state_graph import StateGraph
from pyagenity.graph.state.agent_state import AgentState
from pyagenity.graph.utils.message import Message


def create_string_graph():
    """Create a graph with string response."""

    def string_agent(state, config, checkpointer=None, store=None):
        return "This is a string response that will be streamed character by character!"

    graph = StateGraph()
    graph.add_node("string_node", string_agent)
    graph.set_entry_point("string_node")
    return graph.compile()


def create_dict_graph():
    """Create a graph with dict response."""

    def dict_agent(state, config, checkpointer=None, store=None):
        return {
            "role": "assistant",
            "content": "This is a dict response that will also be streamed smoothly.",
        }

    graph = StateGraph()
    graph.add_node("dict_node", dict_agent)
    graph.set_entry_point("dict_node")
    return graph.compile()


def create_state_graph():
    """Create a graph with AgentState response."""

    def state_agent(state, config, checkpointer=None, store=None):
        new_state = AgentState()
        new_state.context.append(
            Message.from_text(
                "This is an AgentState response that maintains conversation context.",
                role="assistant",
            )
        )
        return new_state

    graph = StateGraph()
    graph.add_node("state_node", state_agent)
    graph.set_entry_point("state_node")
    return graph.compile()


def create_none_graph():
    """Create a graph with None response."""

    def none_agent(state, config, checkpointer=None, store=None):
        # This agent returns None, but the state is still passed through
        return None

    graph = StateGraph()
    graph.add_node("none_node", none_agent)
    graph.set_entry_point("none_node")
    return graph.compile()


def demo_sync_streaming():
    """Demonstrate synchronous streaming."""
    print("🔄 Synchronous Streaming Demo")
    print("=" * 50)

    # Test with different response types
    test_cases = [
        (create_string_graph, "String response streaming"),
        (create_dict_graph, "Dict response streaming"),
        (create_state_graph, "AgentState response streaming"),
        (create_none_graph, "None response streaming"),
    ]

    for graph_creator, description in test_cases:
        print(f"\n📝 {description}:")
        print("-" * 30)

        compiled_graph = graph_creator()
        input_data = {"messages": [Message.from_text(f"Test {description}")]}
        config = {
            "thread_id": f"test_{description.lower().replace(' ', '_')}",
            "recursion_limit": 3,
        }

        chunks = []
        for chunk in compiled_graph.stream(input_data, config=config):
            print(f"  CHUNK: '{str(chunk.content)[:50]}...' (final: {chunk.is_final})")
            chunks.append(chunk)

            if chunk.is_final:
                break

        print(f"  📊 Total chunks: {len(chunks)}")
        if chunks:
            final_content = chunks[-1].content
            print(f"  🎯 Final result: {type(final_content).__name__}")


async def demo_async_streaming():
    """Demonstrate asynchronous streaming."""
    print("\n\n⚡ Asynchronous Streaming Demo")
    print("=" * 50)

    compiled_graph = create_string_graph()

    # Test async streaming
    print("\n📝 Async string response streaming:")
    print("-" * 30)

    input_data = {"messages": [Message.from_text("Async streaming test")]}
    config = {"thread_id": "async_test", "recursion_limit": 3}

    chunks = []
    async for chunk in compiled_graph.astream(input_data, config=config):
        print(f"  ASYNC CHUNK: '{str(chunk.content)[:50]}...' (final: {chunk.is_final})")
        chunks.append(chunk)

        if chunk.is_final:
            break

    print(f"  📊 Total async chunks: {len(chunks)}")


if __name__ == "__main__":
    print("🚀 PyAgenity Streaming Functionality Demo")
    print("=" * 60)
    print("This demo shows the new stream() and astream() methods")
    print("working with different response types from agent functions.")
    print("=" * 60)

    try:
        # Run sync demo
        demo_sync_streaming()

        # Run async demo
        asyncio.run(demo_async_streaming())

        print("\n" + "=" * 60)
        print("✅ All streaming demos completed successfully!")
        print("🎉 The stream and astream methods are ready for use!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise
