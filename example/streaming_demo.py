"""
Example demonstrating streaming functionality in PyAgenity.
Shows both streaming and non-streaming responses.
"""

from typing import Any
from dotenv import load_dotenv
from litellm import completion, acompletion
from pyagenity.graph.graph import StateGraph
from pyagenity.graph.graph.tool_node import ToolNode
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import END, InjectState, InjectToolCallID, Message, convert_messages


# --- Native async streaming node using litellm.acompletion ---
async def native_async_streaming_agent(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
):
    prompts = """
        You are a helpful assistant that provides detailed, informative responses.
        When providing weather information, be descriptive and helpful.
    """
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )
    # Use OpenAI GPT-3.5-turbo for demonstration; ensure your API key is set
    response = await acompletion(model="gpt-3.5-turbo", messages=messages, stream=True)
    return response


def create_native_async_streaming_graph():
    """Create a graph with a node that uses litellm.acompletion (native async streaming)."""
    graph = StateGraph()
    graph.add_node("MAIN", native_async_streaming_agent)
    graph.set_entry_point("MAIN")
    return graph.compile()


async def test_native_async_streaming():
    """Test native async streaming with litellm.acompletion inside a graph node."""
    print("\n=== Testing Native Async Streaming (litellm.acompletion) ===")
    app = create_native_async_streaming_graph()
    inp = {"messages": [Message.from_text("Hello, how are you?")]}
    config = {"thread_id": "native_async_test", "recursion_limit": 3}
    print("\nNative async streaming output:")
    try:
        async for chunk in app.astream(inp, config=config):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
            if chunk.is_final:
                print("\n[Native async stream completed]")
                break
    except Exception as e:
        print(f"\nError in native async streaming: {e}")


from pyagenity.graph.graph import StateGraph
from pyagenity.graph.graph.tool_node import ToolNode
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import END, InjectState, InjectToolCallID, Message, convert_messages


load_dotenv()


def get_weather(
    location: str,
    tool_call_id: InjectToolCallID,
    state: InjectState,
) -> str:
    """
    Get the current weather for a specific location.
    This demo shows injectable parameters: tool_call_id and state are automatically injected.
    """
    # You can access injected parameters here
    if tool_call_id:
        print(f"Tool call ID: {tool_call_id}")
    if state and hasattr(state, "context"):
        print(f"Number of messages in context: {len(state.context)}")

    return f"The weather in {location} is sunny with a temperature of 75°F and light winds."


tool_node = ToolNode([get_weather])


def streaming_agent(
    state: AgentState,
    config: dict[str, Any],
    checkpointer: Any | None = None,
    store: Any | None = None,
):
    """Agent that uses streaming responses."""
    prompts = """
        You are a helpful assistant that provides detailed, informative responses.
        When providing weather information, be descriptive and helpful.
    """

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )

    # Check if the last message is a tool result - if so, make final response without tools
    if (
        state.context
        and len(state.context) > 0
        and state.context[-1].role == "tool"
        and state.context[-1].tool_call_id is not None
    ):
        # Make final response with streaming enabled
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            stream=True,  # Enable streaming
        )
    else:
        # Regular response with tools available and streaming enabled
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tool_node.all_tools(),
            stream=True,  # Enable streaming
        )

    return response


def non_streaming_agent(
    state: AgentState,
    config: dict[str, Any],
    checkpointer: Any | None = None,
    store: Any | None = None,
):
    """Agent that uses non-streaming responses."""
    prompts = """
        You are a helpful assistant that provides concise, direct responses.
    """

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )

    # Check if the last message is a tool result - if so, make final response without tools
    if (
        state.context
        and len(state.context) > 0
        and state.context[-1].role == "tool"
        and state.context[-1].tool_call_id is not None
    ):
        # Make final response without streaming
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            # No stream parameter - will return complete response
        )
    else:
        # Regular response with tools available, no streaming
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tool_node.all_tools(),
            # No stream parameter - will return complete response
        )

    return response


def should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"  # No context, might need tools

    last_message = state.context[-1]

    # If the last message is from assistant and has tool calls, go to TOOL
    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    # If last message is a tool result, we should be done (AI will make final response)
    if last_message.role == "tool" and last_message.tool_call_id is not None:
        return END

    # Default to END for other cases
    return END


def create_streaming_graph():
    """Create a graph that uses streaming responses."""
    graph = StateGraph()
    graph.add_node("MAIN", streaming_agent)
    graph.add_node("TOOL", tool_node)

    # Add conditional edges from MAIN
    graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})

    # Always go back to MAIN after TOOL execution
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")

    return graph.compile()


def create_non_streaming_graph():
    """Create a graph that uses non-streaming responses."""
    graph = StateGraph()
    graph.add_node("MAIN", non_streaming_agent)
    graph.add_node("TOOL", tool_node)

    # Add conditional edges from MAIN
    graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})

    # Always go back to MAIN after TOOL execution
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")

    return graph.compile()


def test_streaming():
    """Test streaming functionality."""
    print("=== Testing Streaming Responses ===")

    app = create_streaming_graph()
    inp = {"messages": [Message.from_text("What's the weather like in New York?")]}
    config = {"thread_id": "streaming_test", "recursion_limit": 10}

    print("\\nStreaming output:")
    try:
        for chunk in app.stream(inp, config=config):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
            if chunk.is_final:
                print("\\n[Stream completed]")
                break
    except Exception as e:
        print(f"\\nError in streaming: {e}")


def test_non_streaming():
    """Test non-streaming functionality with simulated streaming."""
    print("\\n=== Testing Non-Streaming Responses (Simulated Streaming) ===")

    app = create_non_streaming_graph()
    inp = {"messages": [Message.from_text("What's the weather like in Boston?")]}
    config = {"thread_id": "non_streaming_test", "recursion_limit": 10}

    print("\\nSimulated streaming output:")
    try:
        for chunk in app.stream(inp, config=config):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
            if chunk.is_final:
                print("\\n[Stream completed]")
                break
    except Exception as e:
        print(f"\\nError in simulated streaming: {e}")


async def test_async_streaming():
    """Test async streaming functionality."""
    print("\\n=== Testing Async Streaming ===")

    app = create_streaming_graph()
    inp = {"messages": [Message.from_text("What's the weather like in San Francisco?")]}
    config = {"thread_id": "async_streaming_test", "recursion_limit": 10}

    print("\\nAsync streaming output:")
    try:
        async for chunk in app.astream(inp, config=config):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
            if chunk.is_final:
                print("\\n[Async stream completed]")
                break
    except Exception as e:
        print(f"\\nError in async streaming: {e}")


if __name__ == "__main__":
    import asyncio

    # Test both streaming and non-streaming
    test_streaming()
    test_non_streaming()

    # Test async streaming
    asyncio.run(test_async_streaming())

    # Test native async streaming (litellm.acompletion)
    asyncio.run(test_native_async_streaming())

    print("\\n=== All streaming tests completed ===")
