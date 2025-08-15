"""Enhanced simple_graph.py example showing proper type hints with injectable parameters."""

from typing import Any, TypedDict

from dotenv import load_dotenv
from litellm import completion

from pyagenity.graph.graph import StateGraph
from pyagenity.graph.graph.tool_node import ToolNode
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import END, InjectState, InjectToolCallID, Message, convert_messages


load_dotenv()


# Example of a typed config for better IDE support
class GraphConfig(TypedDict):
    """Configuration with proper typing for IDE support."""

    thread_id: str
    recursion_limit: int


def get_weather(
    location: str,
    tool_call_id: InjectToolCallID[str] = None,  # IDE knows: str
    state: InjectState[AgentState] = None,  # IDE knows: AgentState
) -> str:
    """
    Get the current weather for a specific location.

    Now with enhanced type hints:
    - tool_call_id: IDE knows this is str
    - state: IDE knows this is AgentState and provides autocomplete for state.context, etc.
    """
    # IDE will provide autocomplete for these!
    if tool_call_id:
        print(f"Tool call ID: {tool_call_id}")  # IDE knows tool_call_id is str

    if state and hasattr(state, "context"):
        # IDE knows state.context exists and can provide autocomplete
        print(f"Number of messages in context: {len(state.context)}")

    return f"The weather in {location} is sunny."


tool_node = ToolNode([get_weather])


def main_agent(
    state: AgentState,
    config: dict[str, Any],
    checkpointer: Any | None = None,
    store: Any | None = None,
):
    prompts = """
        You are a helpful assistant.
        Your task is to assist the user in finding information and answering questions.
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
        # Make final response without tools since we just got tool results
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
        )
    else:
        # Regular response with tools available
        response = completion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tool_node.all_tools(),
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


# Build the graph
graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)

# Add conditional edges from MAIN
graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})

# Always go back to MAIN after TOOL execution
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")


app = graph.compile()


# Demonstrate the enhanced usage
if __name__ == "__main__":
    print("Enhanced Injectable Types Example")
    print("=" * 40)

    # Show the tool specification - injectable params are excluded
    tools = tool_node.all_tools()
    print("\nTool specification (injectable params excluded):")
    for tool in tools:
        func_name = tool["function"]["name"]
        params = list(tool["function"]["parameters"]["properties"].keys())
        print(f"  {func_name}: {params}")

    print("\nBenefits of enhanced injectable types:")
    print("✓ IDE provides autocomplete for state.context")
    print("✓ IDE knows tool_call_id is str type")
    print("✓ Type safety at development time")
    print("✓ Clean tool specifications for LLMs")

    # Run the actual example
    print("\nRunning graph example...")
    inp = {
        "messages": [Message.from_text("Please call the get_weather function for New York City")]
    }
    config: GraphConfig = {"thread_id": "12345", "recursion_limit": 10}

    res = app.invoke(inp, config=config)
    print(f"\nResult: {len(res['messages'])} messages generated")
