
# PyAgenity

![PyPI](https://img.shields.io/pypi/v/pyagenity?color=blue)
![License](https://img.shields.io/github/license/Iamsdt/pyagenity)
![Python](https://img.shields.io/pypi/pyversions/pyagenity)
[![Coverage](https://img.shields.io/badge/coverage-63%25-yellow.svg)](#)

**PyAgenity** is a lightweight Python framework for building intelligent agents and orchestrating multi-agent workflows, can be used with any LLM provider like OpenAI, Google Gemini, Anthropic Claude, etc.

PyAgenity is a lightweight Python framework for building intelligent agents and orchestrating multi-agent workflows on top of the LiteLLM unified LLM interface.

---

## Features


- Unified `Agent` abstraction (no raw LiteLLM objects leaked)
- Structured responses with `content`, optional `thinking`, and `usage`
- Streaming support with incremental chunks (Delta = True)
- Final message hooks for persistence/logging
- LangGraph-inspired graph engine: nodes, conditional edges, pause/resume (human-in-loop)
- In-memory session state store (pluggable in the future)


---

## Installation

**Basic installation with [uv](https://github.com/astral-sh/uv) (recommended):**

```bash
uv pip install pyagenity
```

Or with pip:

```bash
pip install pyagenity
```

**Optional Dependencies:**

PyAgenity supports optional dependencies for specific functionality:

```bash
# PostgreSQL + Redis checkpointing
pip install pyagenity[pg_checkpoint]

# MCP (Model Context Protocol) support
pip install pyagenity[mcp]

# Composio tools (adapter)
pip install pyagenity[composio]

# LangChain tools (registry-based adapter)
 pip install pyagenity[langchain]

# Individual publishers
pip install pyagenity[redis]     # Redis publisher
pip install pyagenity[kafka]     # Kafka publisher
pip install pyagenity[rabbitmq]  # RabbitMQ publisher

# Multiple extras
pip install pyagenity[pg_checkpoint,mcp,composio,langchain]
```

---

## Development vs. Library Usage

**Library consumers:**
- Only `pyproject.toml` is needed. It contains runtime dependencies and build metadata.
- Install with pip or uv as shown above.

**Development contributors:**
- Use `pyproject.dev.toml` for all development tools, linters, test runners, and optional extras.
- Install dev dependencies with:

```bash
# Option 1: pip (from requirements-dev.txt)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Option 2: pip (editable install with extras, if supported)
pip install -e .[dev]

# Option 3: uv (if you use uv)
uv pip install -r requirements-dev.txt
```

**Note:**
- `pyproject.dev.toml` contains all dev/test/docs/mail extras and tool configs (ruff, isort, mypy, pytest, bandit, etc.).
- `pyproject.toml` is minimal and safe for use as a library dependency in other projects.

Set provider API keys (example for OpenAI):

```bash
export OPENAI_API_KEY=sk-...  # required for gpt-* models
```

If you have a `.env` file, it will be auto-loaded (via `python-dotenv`).
---

---

See `example/graph_demo.py` for a runnable example.

### Using LangChain tools

The LangChain adapter is registry-based. You can register any LangChain tools (BaseTool/StructuredTool or duck-typed run/_run objects), and PyAgenity will expose them to the LLM via a uniform function-calling schema.

```python
from pyagenity.adapters.tools.langchain_adapter import LangChainAdapter
from pyagenity.graph import ToolNode

adapter = LangChainAdapter()  # autoloads a couple common tools if none registered
# Optionally register your own tools:
# from langchain_community.tools import DuckDuckGoSearchRun
# adapter.register_tool(DuckDuckGoSearchRun())

tool_node = ToolNode([], langchain_adapter=adapter)
tools = tool_node.all_tools_sync()  # pass these to your LLM as tools
```

Disable autoload and register explicitly:

```python
adapter = LangChainAdapter(autoload_default_tools=False)
adapter.register_tools([my_tool, another_tool])
```

## Example: React Weather Agent

This repository includes a comprehensive example agent that demonstrates tool injection, conditional edges, and streaming support. The example is in `examples/react/react_weather_agent.py` and uses an in-memory checkpointer. It's intended as a runnable demo showing how to register ToolNodes, conditional edges, and invoke the compiled graph.

Key points:
- Demonstrates an injectable tool signature that receives `tool_call_id` and `state`.
- Shows how to add `ToolNode` and conditional edges into a `StateGraph`.
- Uses `litellm.acompletion` to call an LLM with async support.
- Includes proper error handling and tool call routing.

Complete example from `examples/react/react_weather_agent.py`:

```python
from dotenv import load_dotenv
from litellm import acompletion

from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import StateGraph, ToolNode
from pyagenity.state.agent_state import AgentState
from pyagenity.utils import Message
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages

load_dotenv()

checkpointer = InMemoryCheckpointer()

def get_weather(
    location: str,
    tool_call_id: str | None = None,
    state: AgentState | None = None,
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

async def main_agent(state: AgentState):
    prompts = """
        You are a helpful assistant.
        Your task is to assist the user in finding information and answering questions.
    """

    messages = convert_messages(
        system_prompts=[
            {
                "role": "system",
                "content": prompts,
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": "3600s",  # 👈 Cache for 1 hour
                },
            },
            {"role": "user", "content": "Today Date is 2024-06-15"},
        ],
        state=state,
    )

    mcp_tools = []

    # Check if the last message is a tool result - if so, make final response without tools
    if (
        state.context
        and len(state.context) > 0
        and state.context[-1].role == "tool"
        and state.context[-1].tool_call_id is not None
    ):
        # Make final response without tools since we just got tool results
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
        )
    else:
        # Regular response with tools available
        tools = await tool_node.all_tools()
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tools + mcp_tools,
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

# Run the agent
inp = {"messages": [Message.from_text("Please call the get_weather function for New York City")]}
config = {"thread_id": "12345", "recursion_limit": 10}

res = app.invoke(inp, config=config)

for i in res["messages"]:
    print("**********************")
    print("Message Type: ", i.role)
    print(i)
    print("**********************")
    print("\n\n")
```

How to run the example locally

1. Install dependencies (recommended in a virtualenv):

```bash
pip install -r requirements.txt
# or if you use uv
uv pip install -r requirements.txt
```

2. Set your LLM provider API key (for example OpenAI):

```bash
export OPENAI_API_KEY="sk-..."
# or create a .env with the key and the script will load it automatically
```

3. Run the example script:

```bash
python examples/react/react_weather_agent.py
```

Notes
- The example uses `litellm`'s `acompletion` function — set `model` to a provider/model available in your environment (for example `gemini/gemini-2.5-flash` or other supported model strings).
- `InMemoryCheckpointer` is for demo/testing only. Replace with a persistent checkpointer for production.

---

## Example: MCP Integration

PyAgenity supports integration with Model Context Protocol (MCP) servers, allowing you to connect external tools and services. The example in `examples/react-mcp/` demonstrates how to integrate MCP tools with your agent.

First, create an MCP server (see `examples/react-mcp/server.py`):

```python
from fastmcp import FastMCP

mcp = FastMCP("My MCP Server")

@mcp.tool(
    description="Get the weather for a specific location",
)
def get_weather(location: str) -> dict:
    return {
        "location": location,
        "temperature": "22°C",
        "description": "Sunny",
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

Then, integrate MCP tools into your agent (from `examples/react-mcp/react-mcp.py`):

```python
from typing import Any

from dotenv import load_dotenv
from fastmcp import Client
from litellm import acompletion

from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import StateGraph, ToolNode
from pyagenity.state.agent_state import AgentState
from pyagenity.utils import Message
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages

load_dotenv()

checkpointer = InMemoryCheckpointer()

config = {
    "mcpServers": {
        "weather": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "streamable-http",
        },
        "github": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "streamable-http",
        },
    }
}

client_http = Client(config)

# Initialize ToolNode with MCP client
tool_node = ToolNode(functions=[], client=client_http)

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

    # Get all available tools (including MCP tools)
    tools = await tool_node.all_tools()
    print("**** List of tools", len(tools), tools)

    response = await acompletion(
        model="gemini/gemini-2.0-flash",
        messages=messages,
        tools=tools,
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

# Run the agent
inp = {"messages": [Message.from_text("Please call the get_weather function for New York City")]}
config = {"thread_id": "12345", "recursion_limit": 10}

res = app.invoke(inp, config=config)

for i in res["messages"]:
    print(i)
    print("\n")
```

How to run the MCP example:

1. Install MCP dependencies:
```bash
pip install pyagenity[mcp]
# or
uv pip install pyagenity[mcp]
```

2. Start the MCP server in one terminal:
```bash
cd examples/react-mcp
python server.py
```

3. Run the MCP-integrated agent in another terminal:
```bash
python examples/react-mcp/react-mcp.py
```

---

## Example: Streaming Agent

PyAgenity supports streaming responses for real-time interaction. The example in `examples/react_stream/stream_react_agent.py` demonstrates different streaming modes and configurations.

```python
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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()
checkpointer = InMemoryCheckpointer()

def get_weather(
    location: str,
    tool_call_id: str,
    state: AgentState,
) -> Message:
    """Get weather with injectable parameters."""
    if tool_call_id:
        logging.debug("[tool] Tool call ID: %s", tool_call_id)
    if state and hasattr(state, "context"):
        logging.debug("[tool] Context messages: %s", len(state.context))

    res = f"The weather in {location} is sunny."
    return Message.tool_message(
        content=res,
        tool_call_id=tool_call_id,
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

app = graph.compile(checkpointer=checkpointer)

async def run_stream_test() -> None:
    inp = {"messages": [Message.from_text("Call get_weather for Tokyo, then reply.")]}
    config = {"thread_id": "stream-1", "recursion_limit": 10}

    logging.info("--- streaming start ---")
    stream_gen = app.astream(
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
```

Run the streaming example:
```bash
python examples/react_stream/stream_react_agent.py
```

Or run specific test modes:
```bash
python examples/react_stream/stream_react_agent.py sync
python examples/react_stream/stream_react_agent.py non-stream
python examples/react_stream/stream_react_agent.py sync-stream
```

---

## Stopping a running graph

You can cooperatively stop an in-flight execution (invoke or stream) from your UI/frontend by setting a stop flag in the checkpointer’s thread store. Handlers poll this flag and gracefully interrupt the run.

Contract:
- Provide a stable `thread_id` in the `config` for both the running graph and the stop request
- Call `app.stop(config)` (sync) or `await app.astop(config)` (async)
- The stop method returns a small status dict with `ok`, `running`, and `thread` info

Minimal example:

```python
from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import StateGraph
from pyagenity.state.agent_state import AgentState
from pyagenity.utils import Message
from pyagenity.utils.constants import END

checkpointer = InMemoryCheckpointer()

async def main_agent(state: AgentState, config: dict | None = None):
    # Produce chunks (streaming) or any long-running work
    import asyncio
    for i in range(10):
        await asyncio.sleep(0.3)
        yield Message.text_message(f"Chunk {i+1}")

def router(state: AgentState) -> str:
    return END

graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_conditional_edges("MAIN", router, {END: END})
graph.set_entry_point("MAIN")
app = graph.compile(checkpointer=checkpointer)

config = {"thread_id": "demo-thread", "is_stream": True}
inp = {"messages": [Message.text_message("Start, then stop soon")]}

# Start streaming in a background thread/task and then request stop:
import threading, time
def reader():
    for chunk in app.stream(inp, config=config):
        print("STREAM:", getattr(chunk, "content", chunk))

t = threading.Thread(target=reader, daemon=True)
t.start()
time.sleep(1.0)
print("Stop status:", app.stop(config))
t.join(timeout=5)
```

See runnable examples in `examples/react_stream/stop_stream.py` and `examples/react_stream/stop_stream_litellm.py`.

---

## Roadmap


- Persistent state backend (Redis, SQL, etc.)
- Parallel / branching strategies and selection policies
- Tool invocation nodes & function calling wrappers
- Tracing / telemetry integration

---

## TODO

- **Stop Current Execution**: Allow stopping graph execution from frontend/UI
- **Remote Node Support**: Enable running nodes on remote machines for distributed processing
- **Extend Node Support**: Allow users to extend and customize Node and ToolNode classes

---

## License

MIT

---

## Project Links

- [GitHub Repository](https://github.com/Iamsdt/pyagenity)
- [PyPI Project Page](https://pypi.org/project/pyagenity/)

---