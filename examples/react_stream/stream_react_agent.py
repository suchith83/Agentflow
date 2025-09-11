import asyncio
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


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

# Debug: Print registered tools
print("Registered tools:", list(tool_node._funcs.keys()))


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


def main_agent_sync(
    state: AgentState,
    config: dict[str, Any],
    checkpointer: Any | None = None,
    store: Any | None = None,
):
    """
    Synchronous version of main_agent for testing.
    """

    prompts = """
        You are a helpful assistant.
        Answer conversationally. Use tools when needed.
    """

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )

    is_stream = config.get("is_stream", False)

    async def _async_call():
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
                stream=False,
            )
        return response

    # Run the async call synchronously
    return asyncio.run(_async_call())


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
        print(chunk.model_dump(), end="\n", flush=True)


async def run_sync_test() -> None:
    """Test sync main_agent implementation"""
    # Create a graph with sync main_agent
    sync_graph = StateGraph()
    sync_graph.add_node("MAIN", main_agent_sync)
    sync_graph.add_node("TOOL", tool_node)

    sync_graph.add_conditional_edges(
        "MAIN",
        should_use_tools,
        {"TOOL": "TOOL", END: END},
    )

    sync_graph.add_edge("TOOL", "MAIN")
    sync_graph.set_entry_point("MAIN")

    sync_app = sync_graph.compile(
        checkpointer=checkpointer,
    )

    inp = {"messages": [Message.from_text("Call get_weather for Tokyo, then reply.")]}
    config = {"thread_id": "sync-1", "recursion_limit": 10}

    logging.info("--- sync test start ---")
    stream_gen = await sync_app.astream(
        inp,
        config=config,
        response_granularity=ResponseGranularity.LOW,
    )
    async for chunk in stream_gen:
        print(chunk.model_dump(), end="\n", flush=True)


async def run_sync_stream_test() -> None:
    """Test sync stream main_agent implementation"""
    # Create a graph with sync stream main_agent
    sync_stream_graph = StateGraph()

    def main_agent_sync_stream(
        state: AgentState,
        config: dict[str, Any],
        checkpointer: Any | None = None,
        store: Any | None = None,
    ):
        """
        Synchronous streaming version of main_agent for testing.
        """

        prompts = """
            You are a helpful assistant.
            Answer conversationally. Use tools when needed.
        """

        messages = convert_messages(
            system_prompts=[{"role": "system", "content": prompts}],
            state=state,
        )

        # Enable streaming
        is_stream = True

        async def _async_call():
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
                response = await acompletion(
                    model="gemini/gemini-2.5-flash",
                    messages=messages,
                    tools=tools,
                    stream=is_stream,
                )
            return response

        # Run the async call synchronously
        return asyncio.run(_async_call())

    sync_stream_graph.add_node("MAIN", main_agent_sync_stream)
    sync_stream_graph.add_node("TOOL", tool_node)

    sync_stream_graph.add_conditional_edges(
        "MAIN",
        should_use_tools,
        {"TOOL": "TOOL", END: END},
    )

    sync_stream_graph.add_edge("TOOL", "MAIN")
    sync_stream_graph.set_entry_point("MAIN")

    sync_stream_app = sync_stream_graph.compile(
        checkpointer=checkpointer,
    )

    inp = {"messages": [Message.from_text("Call get_weather for Tokyo, then reply.")]}
    config = {"thread_id": "sync-stream-1", "recursion_limit": 10}

    logging.info("--- sync stream test start ---")
    stream_gen = await sync_stream_app.astream(
        inp,
        config=config,
        response_granularity=ResponseGranularity.LOW,
    )
    async for chunk in stream_gen:
        print(chunk.model_dump(), end="\n", flush=True)


async def run_non_stream_test() -> None:
    """Test non-streaming main_agent implementation"""
    # Create a graph with non-streaming main_agent
    non_stream_graph = StateGraph()

    async def main_agent_non_stream(
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

        # Always disable streaming
        is_stream = False

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
            response = await acompletion(
                model="gemini/gemini-2.5-flash",
                messages=messages,
                tools=tools,
                stream=is_stream,
            )

        return response

    non_stream_graph.add_node("MAIN", main_agent_non_stream)
    non_stream_graph.add_node("TOOL", tool_node)

    non_stream_graph.add_conditional_edges(
        "MAIN",
        should_use_tools,
        {"TOOL": "TOOL", END: END},
    )

    non_stream_graph.add_edge("TOOL", "MAIN")
    non_stream_graph.set_entry_point("MAIN")

    non_stream_app = non_stream_graph.compile(
        checkpointer=checkpointer,
    )

    inp = {"messages": [Message.from_text("Call get_weather for Tokyo, then reply.")]}
    config = {"thread_id": "non-stream-1", "recursion_limit": 10}

    logging.info("--- non-stream test start ---")
    stream_gen = await non_stream_app.astream(
        inp,
        config=config,
        response_granularity=ResponseGranularity.LOW,
    )
    async for chunk in stream_gen:
        print(chunk.model_dump(), end="\n", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "sync":
            asyncio.run(run_sync_test())
        elif test_type == "non-stream":
            asyncio.run(run_non_stream_test())
        elif test_type == "sync-stream":
            asyncio.run(run_sync_stream_test())
        else:
            logging.info("Usage: python stream_react_agent.py [sync|non-stream|sync-stream]")
            logging.info("Running default streaming test...")
            asyncio.run(run_stream_test())
    else:
        asyncio.run(run_stream_test())
