from typing import Any
from dataclasses import field

from dotenv import load_dotenv
from litellm import acompletion

from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import StateGraph, ToolNode
from pyagenity.state.agent_state import AgentState
from pyagenity.utils import Message
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages
from pyagenity.utils.injectable import InjectState, InjectToolCallID


load_dotenv()


class MyState(AgentState):
    """Custom state with additional fields for resume matching."""

    candidate_cv: str = ""
    jd: str = ""  # job description
    match_score: float = 0.0
    analysis_results: dict = field(default_factory=dict)


# Create checkpointer with custom state type
checkpointer = InMemoryCheckpointer[MyState]()


async def main_agent(
    state: MyState,
    config: dict[str, Any],
    checkpointer: Any | None = None,
    store: Any | None = None,
):
    """Main agent that processes CV and job description."""
    prompts = f"""
        You are a helpful HR assistant.
        Your task is to analyze candidate CVs against job descriptions.
        
        Current state:
        - Candidate CV: {state.candidate_cv[:100]}...
        - Job Description: {state.jd[:100]}...
        - Previous match score: {state.match_score}
        
        Please provide a helpful response to the user's query.
    """

    print(f"Processing state with CV length: {len(state.candidate_cv)}")
    print(f"Job Description length: {len(state.jd)}")
    print(f"Current match score: {state.match_score}")

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )

    response = await acompletion(
        model="gemini/gemini-2.5-flash",
        messages=messages,
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


# Create graph with custom state type
graph = StateGraph[MyState](MyState())
graph.add_node("MAIN", main_agent)
# graph.add_node("TOOL", tool_node)

# Add conditional edges from MAIN
# graph.add_conditional_edges(
#     "MAIN",
#     should_use_tools,
#     {"TOOL": "TOOL", END: END},
# )

# Always go back to MAIN after TOOL execution
# graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")


app = graph.compile(
    checkpointer=checkpointer,
)


def test_basic_functionality():
    """Test basic functionality with default state."""
    print("=== Testing Basic Functionality ===")
    inp = {"messages": [Message.from_text("Hello, can you help me with CV analysis?")]}
    config = {"thread_id": "basic_test", "recursion_limit": 10}

    res = app.invoke(inp, config=config)
    print("Basic test result:", res)
    return res


def test_custom_state_fields():
    """Test with custom state fields populated."""
    print("\n=== Testing Custom State Fields ===")

    # Create a custom state with some data
    custom_state = MyState()
    custom_state.candidate_cv = "John Doe - Software Engineer with 5 years Python experience"
    custom_state.jd = "Looking for Senior Python Developer with 3+ years experience"
    custom_state.match_score = 0.85
    custom_state.analysis_results = {"skills_match": True, "experience_match": True}

    # Create new graph with populated state
    custom_graph = StateGraph[MyState](custom_state)
    custom_graph.add_node("MAIN", main_agent)
    custom_graph.set_entry_point("MAIN")

    custom_app = custom_graph.compile(checkpointer=checkpointer)

    inp = {"messages": [Message.from_text("What's the match score for this candidate?")]}
    config = {"thread_id": "custom_test", "recursion_limit": 10}

    res = custom_app.invoke(inp, config=config)
    print("Custom state test result:", res)
    return res


if __name__ == "__main__":
    # Run tests
    try:
        test_basic_functionality()
        test_custom_state_fields()
        print("\n=== All tests completed successfully! ===")
    except Exception as e:
        print(f"Error during testing: {e}")
        raise
