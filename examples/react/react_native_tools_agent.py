"""Demo: Using native tools with ToolNode in a simple React-style flow.

This example builds a ToolNode from pyagenity's native tools and wires it into a
simple StateGraph similar to the react_weather_agent example.

Note: You'll need an LLM provider configured for litellm if you want to run it end-to-end.
"""

import logging

from litellm import acompletion

from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import StateGraph, ToolNode
from pyagenity.prebuilt.tool import filter_tools, get_native_tools
from pyagenity.state.agent_state import AgentState
from pyagenity.utils import Message
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages


checkpointer = InMemoryCheckpointer()


# Build a ToolNode with only HTTP and files tools enabled
def build_tool_node() -> ToolNode:
    native_tools = get_native_tools(sandbox_root=".sandbox")
    # keep http_get + file tools
    selected = filter_tools(
        native_tools,
        capabilities=["network", "files"],
    )
    return ToolNode(selected)


tool_node = build_tool_node()


async def main_agent(state: AgentState):
    system = "You can decide to use tools for web lookups or file operations when helpful."
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": system}],
        state=state,
    )

    # If we just received tool results, finalize without tools
    if state.context and state.context[-1].role == "tool":
        return await acompletion(model="gemini/gemini-2.5-flash", messages=messages)

    tools = await tool_node.all_tools()
    return await acompletion(model="gemini/gemini-2.5-flash", messages=messages, tools=tools)


def should_use_tools(state: AgentState) -> str:
    if not state.context:
        return "TOOL"
    last = state.context[-1]
    if getattr(last, "tools_calls", None) and last.role == "assistant":
        return "TOOL"
    if last.role == "tool":
        return "MAIN"
    return END


graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)
graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    inp = {
        "messages": [Message.from_text("Fetch https://example.com and save it to folder/page.txt")]
    }
    cfg = {"thread_id": "demo-1", "recursion_limit": 10}
    res = app.invoke(inp, config=cfg)
    for m in res["messages"]:
        logging.info("--- %s ---\n%s", m.role, m.content)
