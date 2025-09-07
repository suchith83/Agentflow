import logging
from typing import Any

from dotenv import load_dotenv
from litellm import acompletion

from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import StateGraph, ToolNode
from pyagenity.state.agent_state import AgentState
from pyagenity.utils import Message, ResponseGranularity
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages


load_dotenv()

checkpointer = InMemoryCheckpointer()


def get_weather(
    location: str,
    tool_call_id: str,
    state: AgentState,
) -> Message:
    """
    Get the current weather for a specific location.
    Demonstrates injectable parameters: tool_call_id and state are automatically injected.
    """
    if tool_call_id:
        logging.debug("[tool] Tool call ID: %s", tool_call_id)
    if state and hasattr(state, "context"):
        logging.debug("[tool] Context messages: %s", len(state.context))  # type: ignore

    res = f"The weather in {location} is sunny."
    return Message.tool_message(
        content=res,
        tool_call_id=tool_call_id,  # type: ignore
    )


tool_node = ToolNode([get_weather])


async def main_agent(
    state: AgentState,
    config: dict[str, Any],
    checkpointer: Any | None = None,
    store: Any | None = None,
):
    prompts = """
        You are a helpful assistant.
        Answer conversationally. Use tools when needed.
    """

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )

    is_stream = config.get("is_stream", False)

    if (
        state.context
        and len(state.context) > 0
        and state.context[-1].role == "tool"
        and state.context[-1].tool_call_id is not None
    ):
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            stream=is_stream,
        )
    else:
        tools = await tool_node.all_tools()
        # Avoid streaming when tools are enabled to ensure tool-calls are parsed properly
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tools,
            stream=is_stream,
        )

    return response


def should_use_tools(state: AgentState) -> str:
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    if last_message.role == "tool" and last_message.tool_call_id is not None:
        return END

    return END


graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)

graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"TOOL": "TOOL", END: END},
)

graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")


app = graph.compile(
    checkpointer=checkpointer,
)


async def run_stream_test() -> None:
    inp = {"messages": [Message.from_text("Call get_weather for Tokyo, then reply.")]}
    config = {"thread_id": "stream-1", "recursion_limit": 10}

    logging.info("--- streaming start ---")
    stream_gen = await app.astream(
        inp,
        config=config,
        response_granularity=ResponseGranularity.LOW,
    )
    async for chunk in stream_gen:
        meta = chunk.meta or {}
        event = meta.get("event")
        run_id = meta.get("run_id")
        msg_id = meta.get("message_id")
        node = meta.get("node")
        step = meta.get("step")
        if event == "delta":
            logging.info(
                "[delta] run=%s node=%s step=%s msg=%s delta=%r",
                run_id,
                node,
                step,
                msg_id,
                chunk.delta,
            )
        else:
            logging.info(
                "[event] %s run=%s node=%s step=%s msg=%s",
                event,
                run_id,
                node,
                step,
                msg_id,
            )
            if event == "graph_completed":
                logging.info("result: %s", meta.get("result"))
    logging.info("--- streaming end ---")


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run_stream_test())
