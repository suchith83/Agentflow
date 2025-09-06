from dotenv import load_dotenv
from injectq import Inject, InjectQ, inject, injectq

from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.checkpointer.base_checkpointer import BaseCheckpointer
from pyagenity.graph import StateGraph, ToolNode
from pyagenity.state.agent_state import AgentState
from pyagenity.store.base_store import BaseStore
from pyagenity.utils import Message
from pyagenity.utils.callbacks import CallbackManager
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages


load_dotenv()

checkpointer = InMemoryCheckpointer()


@inject
def get_weather(
    location: str,
    tool_call_id: str,
    state: AgentState,
    config: dict,
) -> Message:
    """
    Get the current weather for a specific location.
    This demo shows injectable parameters: tool_call_id and state are automatically injected.
    """
    # You can access injected parameters here
    if tool_call_id:
        print(f"Tool call ID: {tool_call_id}")
    if state and hasattr(state, "context"):
        print(f"Number of messages in context: {len(state.context)}")  # type: ignore

    res = f"The weather in {location} is sunny"
    return Message.tool_message(
        content=res,
        tool_call_id=tool_call_id,  # type: ignore
    )


tool_node = ToolNode([get_weather])


async def main_agent(
    # state: AgentState,
    # config: dict,
    callback: CallbackManager = Inject[CallbackManager],
    # checkpointer: InMemoryCheckpointer | None = None,
    # store: BaseStore | None = None,
):
    prompts = """
        You are a helpful assistant.
        Your task is to assist the user in finding information and answering questions.
    """

    # messages = convert_messages(
    #     system_prompts=[{"role": "system", "content": prompts}],
    #     state=state,
    # )

    mcp_tools = []

    # # Check if the last message is a tool result - if so, make final response without tools
    # if (
    #     state.context
    #     and len(state.context) > 0
    #     and state.context[-1].role == "tool"
    #     and state.context[-1].tool_call_id is not None
    # ):
    #     # Make final response without tools since we just got tool results
    #     response = await acompletion(
    #         model="gemini/gemini-2.5-flash",
    #         messages=messages,
    #     )
    # else:
    #     # Regular response with tools available
    #     tools = await tool_node.all_tools()
    #     response = await acompletion(
    #         model="gemini/gemini-2.5-flash",
    #         messages=messages,
    #         tools=tools + mcp_tools,
    #     )

    message_id = injectq.get("generated_id")
    # message_id2 = injectq.try_get("generated_id2", "final-response-579898")
    # print("Generated Message ID: ", message_id)
    # print("Generated Message ID 2: ", message_id2)
    print("checkpointer", checkpointer)
    # print("state", state)
    print("config", config)
    # print("State", len(state.context))
    # print("Store", store)
    # print("Callback", callback)

    return Message(
        message_id="final-response-579898",
        content="This is example final response from main agent.",
        role="assistant",
        tools_calls=[],
    )


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


injectq["generated_id2"] = "main-response-12345"

graph = StateGraph(container=None)
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)

# Add conditional edges from MAIN
graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"TOOL": "TOOL", END: END},
)

# Always go back to MAIN after TOOL execution
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")


app = graph.compile(
    checkpointer=checkpointer,
)


# now run it

inp = {"messages": [Message.from_text("Please call the get_weather function for New York City")]}
config = {"thread_id": "12345", "recursion_limit": 10}

print("***** GRAPH", injectq.get_dependency_graph())
res = app.invoke(inp, config=config)

for i in res["messages"]:
    print("**********************")
    print("Message Type: ", i.role)
    print(i)
    print("**********************")
    print("\n\n")


# grp = app.generate_graph()

# print(grp)
