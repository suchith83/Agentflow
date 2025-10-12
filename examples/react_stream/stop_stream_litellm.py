import logging
import threading
import time

from dotenv import load_dotenv
from litellm import acompletion  # type: ignore

from pyagenity.adapters.llm.model_response_converter import ModelResponseConverter
from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import StateGraph, ToolNode
from pyagenity.state import AgentState, Message
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages


logging.basicConfig(level=logging.INFO)
load_dotenv()

checkpointer = InMemoryCheckpointer()


def get_weather(
    location: str,
    tool_call_id: str | None = None,
    state: AgentState | None = None,
) -> str:
    return f"The weather in {location} is sunny"


tool_node = ToolNode([get_weather])


async def main_agent(
    state: AgentState,
    config: dict | None = None,
):
    config = config or {}
    prompts = """
        You are a helpful assistant.
        Your task is to assist the user in finding information and answering questions.
    """

    messages = convert_messages(
        system_prompts=[
            {
                "role": "system",
                "content": prompts,
            },
            {"role": "user", "content": "Today Date is 2024-06-15"},
        ],
        state=state,
    )

    mcp_tools: list = []
    is_stream = config.get("is_stream", False)

    if state.context and len(state.context) > 0 and state.context[-1].role == "tool":
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            stream=is_stream,
        )
    else:
        tools = await tool_node.all_tools()
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tools + mcp_tools,
            stream=is_stream,
        )

    return ModelResponseConverter(response, converter="litellm")


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
    if last_message.role == "tool":
        return "MAIN"
    return END


def build_app():
    graph = StateGraph()
    graph.add_node("MAIN", main_agent)
    graph.add_node("TOOL", tool_node)
    graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")
    return graph.compile(checkpointer=checkpointer)


def run_and_stop_stream():
    app = build_app()
    inp = {
        "messages": [
            Message.text_message(
                "Please call the get_weather function for New York City and think out loud",
            )
        ]
    }
    config = {"thread_id": "stop-stream-thread", "recursion_limit": 10, "is_stream": True}

    def reader():
        for chunk in app.stream(inp, config=config):
            # Demonstrate streaming output while stop may be requested
            logging.info("STREAM: %s", getattr(chunk, "content", chunk))

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # Let the stream run a bit then request stop
    time.sleep(2.0)
    stop_status = app.stop(config)
    logging.info("Requested stop: %s", stop_status)

    t.join(timeout=10)


if __name__ == "__main__":
    run_and_stop_stream()
