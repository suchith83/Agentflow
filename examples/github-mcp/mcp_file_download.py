import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastmcp import Client
from litellm import acompletion

from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.checkpointer import InMemoryCheckpointer
from agentflow.graph import StateGraph, ToolNode
from agentflow.state import AgentState, Message
from agentflow.utils.constants import END
from agentflow.utils.converter import convert_messages


# Root logger: show INFO and above
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Set agentflow logger to DEBUG explicitly
logging.getLogger("agentflow").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


load_dotenv()

checkpointer = InMemoryCheckpointer()

config = {
    "mcpServers": {
        # "weather": {
        #     "url": "http://127.0.0.1:8000/mcp",
        #     "transport": "streamable-http",
        # },
        "github": {
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"},
            "transport": "streamable-http",
        },
    }
}


client_http = Client(config)


tool_node = ToolNode(tools=[], client=client_http)


async def main_agent(
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
    tools = await tool_node.all_tools()
    # print("**** List of tools", len(tools), tools)
    print("**** List of tools", len(tools))
    response = await acompletion(
        model="gemini/gemini-2.0-flash",
        messages=messages,
        tools=tools,
    )
    return ModelResponseConverter(
        response,
        converter="litellm",
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
    if last_message.role == "tool":
        return END

    # Default to END for other cases
    return END


graph = StateGraph()
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

inp = {
    "messages": [
        Message.text_message(
            "Get Readme.md file form the github repo 'https://github.com/suchith83/portfolio' of the 'suchith83' username,."
        )
    ]
}
# inp = {"messages": [Message.from_text("Please call the get_weather function for New York City")]}
config = {"thread_id": "12345", "recursion_limit": 10}
# todo pass data to state directly
res = app.invoke(inp, config=config)

for i in res["messages"]:
    print("***********************")
    print(i)
    print("\n")
