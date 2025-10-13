
# 10xScale Agentflow

![PyPI](https://img.shields.io/pypi/v/agentflow?color=blue)
![License](https://img.shields.io/github/license/10xhub/agentflow)
![Python](https://img.shields.io/pypi/pyversions/agentflow)
[![Coverage](https://img.shields.io/badge/coverage-74%25-yellow.svg)](#)

**10xScale Agentflow** is a lightweight Python framework for building intelligent agents and orchestrating multi-agent workflows. It's an **LLM-agnostic orchestration tool** that works with any LLM provider—use LiteLLM, native SDKs from OpenAI, Google Gemini, Anthropic Claude, or any other provider. You choose your LLM library; 10xScale Agentflow provides the workflow orchestration.

---

## ✨ Key Features

- **🎯 LLM-Agnostic Orchestration** - Works with any LLM provider (LiteLLM, OpenAI, Gemini, Claude, native SDKs)
- **🤖 Multi-Agent Workflows** - Build complex agent systems with your choice of orchestration patterns
- **📊 Structured Responses** - Get `content`, optional `thinking`, and `usage` in a standardized format
- **🌊 Streaming Support** - Real-time incremental responses with delta updates
- **🔧 Tool Integration** - Native support for function calling, MCP, Composio, and LangChain tools with **parallel execution**
- **🔀 LangGraph-Inspired Engine** - Flexible graph orchestration with nodes, conditional edges, and control flow
- **💾 State Management** - Built-in persistence with in-memory and PostgreSQL+Redis checkpointers
- **🔄 Human-in-the-Loop** - Pause/resume execution for approval workflows and debugging
- **🚀 Production-Ready** - Event publishing (Console, Redis, Kafka, RabbitMQ), metrics, and observability
- **🧩 Dependency Injection** - Clean parameter injection for tools and nodes
- **📦 Prebuilt Patterns** - React, RAG, Swarm, Router, MapReduce, SupervisorTeam, and more

---

## Installation

**Basic installation with [uv](https://github.com/astral-sh/uv) (recommended):**

```bash
uv pip install 10xscale-agentflow
```

Or with pip:

```bash
pip install 10xscale-agentflow
```

**Optional Dependencies:**

10xScale Agentflow supports optional dependencies for specific functionality:

```bash
# PostgreSQL + Redis checkpointing
pip install 10xscale-agentflow[pg_checkpoint]

# MCP (Model Context Protocol) support
pip install 10xscale-agentflow[mcp]

# Composio tools (adapter)
pip install 10xscale-agentflow[composio]

# LangChain tools (registry-based adapter)
pip install 10xscale-agentflow[langchain]

# Individual publishers
pip install 10xscale-agentflow[redis]     # Redis publisher
pip install 10xscale-agentflow[kafka]     # Kafka publisher
pip install 10xscale-agentflow[rabbitmq]  # RabbitMQ publisher

# Multiple extras
pip install 10xscale-agentflow[pg_checkpoint,mcp,composio,langchain]
```

### Environment Setup

Set your LLM provider API key:

```bash
export OPENAI_API_KEY=sk-...  # for OpenAI models
# or
export GEMINI_API_KEY=...     # for Google Gemini
# or
export ANTHROPIC_API_KEY=...  # for Anthropic Claude
```

If you have a `.env` file, it will be auto-loaded (via `python-dotenv`).

--- ## 💡 Simple Example

Here's a minimal React agent with tool calling:

```python
from dotenv import load_dotenv
from litellm import acompletion

from agentflow.checkpointer import InMemoryCheckpointer
from agentflow.graph import StateGraph, ToolNode
from agentflow.state.agent_state import AgentState
from agentflow.utils import Message
from agentflow.utils.constants import END
from agentflow.utils.converter import convert_messages

load_dotenv()


# Define a tool with dependency injection
def get_weather(
        location: str,
        tool_call_id: str | None = None,
        state: AgentState | None = None,
) -> Message:
    """Get the current weather for a specific location."""
    res = f"The weather in {location} is sunny"
    return Message.tool_message(
        content=res,
        tool_call_id=tool_call_id,
    )


# Create tool node
tool_node = ToolNode([get_weather])


# Define main agent node
async def main_agent(state: AgentState):
    prompts = "You are a helpful assistant. Use tools when needed."

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )

    # Check if we need tools
    if (
            state.context
            and len(state.context) > 0
            and state.context[-1].role == "tool"
    ):
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
        )
    else:
        tools = await tool_node.all_tools()
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tools,
        )

    return response


# Define routing logic
def should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end."""
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    if (
            hasattr(last_message, "tools_calls")
            and last_message.tools_calls
            and len(last_message.tools_calls) > 0
    ):
        return "TOOL"

    return END


# Build the graph
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

# Compile and run
app = graph.compile(checkpointer=InMemoryCheckpointer())

inp = {"messages": [Message.from_text("What's the weather in New York?")]}
config = {"thread_id": "12345", "recursion_limit": 10}

res = app.invoke(inp, config=config)

for msg in res["messages"]:
    print(msg)
```

### How to run the example locally

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

Notes:
- The example uses `litellm`'s `acompletion` function — set `model` to a provider/model available in your environment (for example `gemini/gemini-2.5-flash` or other supported model strings).
- `InMemoryCheckpointer` is for demo/testing only. Replace with a persistent checkpointer for production.

---

## Example: MCP Integration

10xScale Agentflow supports integration with Model Context Protocol (MCP) servers, allowing you to connect external tools and services. The example in `examples/react-mcp/` demonstrates how to integrate MCP tools with your agent.

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

from agentflow.checkpointer import InMemoryCheckpointer
from agentflow.graph import StateGraph, ToolNode
from agentflow.state.agent_state import AgentState
from agentflow.utils import Message
from agentflow.utils.constants import END
from agentflow.utils.converter import convert_messages

load_dotenv()

checkpointer = InMemoryCheckpointer()

config = {
    "mcpServers": {
        "weather": {
            "url": "http://127.0.0.1:8000/mcp",
            "transport": "streamable-http",
        },
    }
}

client_http = Client(config)

# Initialize ToolNode with MCP client
tool_node = ToolNode(functions=[], client=client_http)


async def main_agent(state: AgentState):
    prompts = "You are a helpful assistant."

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )

    # Get all available tools (including MCP tools)
    tools = await tool_node.all_tools()

    response = await acompletion(
        model="gemini/gemini-2.0-flash",
        messages=messages,
        tools=tools,
    )
    return response


def should_use_tools(state: AgentState) -> str:
    """Determine if we should use tools or end the conversation."""
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    if (
            hasattr(last_message, "tools_calls")
            and last_message.tools_calls
            and len(last_message.tools_calls) > 0
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

# Run the agent
inp = {"messages": [Message.from_text("Please call the get_weather function for New York City")]}
config = {"thread_id": "12345", "recursion_limit": 10}

res = app.invoke(inp, config=config)

for i in res["messages"]:
    print(i)
```

How to run the MCP example:

1. Install MCP dependencies:
```bash
pip install 10xscale-agentflow[mcp]
# or
uv pip install 10xscale-agentflow[mcp]
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

10xScale Agentflow supports streaming responses for real-time interaction. The example in `examples/react_stream/stream_react_agent.py` demonstrates different streaming modes and configurations.

```python
import asyncio
import logging

from dotenv import load_dotenv
from litellm import acompletion

from agentflow.checkpointer import InMemoryCheckpointer
from agentflow.graph import StateGraph, ToolNode
from agentflow.state.agent_state import AgentState
from agentflow.utils import Message, ResponseGranularity
from agentflow.utils.constants import END
from agentflow.utils.converter import convert_messages

load_dotenv()
checkpointer = InMemoryCheckpointer()


def get_weather(
        location: str,
        tool_call_id: str,
        state: AgentState,
) -> Message:
    """Get weather with injectable parameters."""
    res = f"The weather in {location} is sunny."
    return Message.tool_message(
        content=res,
        tool_call_id=tool_call_id,
    )


tool_node = ToolNode([get_weather])


async def main_agent(state: AgentState, config: dict):
    prompts = "You are a helpful assistant. Answer conversationally. Use tools when needed."

    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )

    is_stream = config.get("is_stream", False)

    if (
            state.context
            and len(state.context) > 0
            and state.context[-1].role == "tool"
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


def should_use_tools(state: AgentState) -> str:
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    if (
            hasattr(last_message, "tools_calls")
            and last_message.tools_calls
            and len(last_message.tools_calls) > 0
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


async def run_stream_test():
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
    asyncio.run(run_stream_test())
```

Run the streaming example:
```bash
python examples/react_stream/stream_react_agent.py
```

---

## ⚡ Parallel Tool Execution

10xScale Agentflow automatically executes multiple tool calls **in parallel** when an LLM requests multiple tools simultaneously. This dramatically improves performance for I/O-bound operations.

### Benefits

- **Faster Response Times**: Multiple API calls execute concurrently
- **Better Resource Utilization**: Don't wait for one tool to finish before starting the next
- **Seamless Integration**: Works automatically with existing code - no changes needed

### Example Performance

```python
# LLM requests 3 tools simultaneously:
# - get_weather("NYC")    # Takes 1.0s
# - get_news("tech")      # Takes 1.5s
# - get_stock("AAPL")     # Takes 0.8s

# Sequential execution: 1.0 + 1.5 + 0.8 = 3.3 seconds
# Parallel execution:   max(1.0, 1.5, 0.8) = 1.5 seconds ⚡
```

See the [parallel tool execution documentation](https://10xhub.github.io/10xScale Agentflow/Concept/graph/tools/#parallel-tool-execution) for more details.

---

## 🎯 Use Cases & Patterns

10xScale Agentflow includes prebuilt agent patterns for common scenarios:

### 🤖 Agent Types

- **React Agent** - Reasoning and acting with tool calls
- **RAG Agent** - Retrieval-augmented generation
- **Guarded Agent** - Input/output validation and safety
- **Plan-Act-Reflect** - Multi-step reasoning

### 🔀 Orchestration Patterns

- **Router Agent** - Route queries to specialized agents
- **Swarm** - Dynamic multi-agent collaboration
- **SupervisorTeam** - Hierarchical agent coordination
- **MapReduce** - Parallel processing and aggregation
- **Sequential** - Linear workflow chains
- **Branch-Join** - Parallel branches with synchronization

### 🔬 Advanced Patterns

- **Deep Research** - Multi-level research and synthesis
- **Network** - Complex agent networks

See the [documentation](https://10xhub.github.io/Agentflow/) for complete examples.

---

## 🔧 Development

### For Library Users

Install 10xScale Agentflow as shown above. The `pyproject.toml` contains all runtime dependencies.

### For Contributors

```bash
# Clone the repository
git clone https://github.com/10xhub/10xScale Agentflow.git
cd 10xScale Agentflow

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dev dependencies
pip install -r requirements-dev.txt
# or
uv pip install -r requirements-dev.txt

# Run tests
make test
# or
pytest -q

# Build docs
make docs-serve  # Serves at http://127.0.0.1:8000

# Run examples
cd examples/react
python react_sync.py
```

### Development Tools

The project uses:
- **pytest** for testing (with async support)
- **ruff** for linting and formatting
- **mypy** for type checking
- **mkdocs** with Material theme for documentation
- **coverage** for test coverage reports

See `pyproject.dev.toml` for complete tool configurations.

---

## 🗺️ Roadmap

- ✅ Core graph engine with nodes and edges
- ✅ State management and checkpointing
- ✅ Tool integration (MCP, Composio, LangChain)
- ✅ **Parallel tool execution** for improved performance
- ✅ Streaming and event publishing
- ✅ Human-in-the-loop support
- ✅ Prebuilt agent patterns
- 🚧 Agent-to-Agent (A2A) communication protocols
- 🚧 Remote node execution for distributed processing
- 🚧 Enhanced observability and tracing
- 🚧 More persistence backends (Redis, DynamoDB)
- 🚧 Parallel/branching strategies
- 🚧 Visual graph editor

---

## 📄 License

MIT License - see [LICENSE](https://github.com/10xhub/10xScale Agentflow/blob/main/LICENSE) for details.

---

## 🔗 Links & Resources

- **[Documentation](https://10xhub.github.io/10xScale Agentflow/)** - Full documentation with tutorials and API reference
- **[GitHub Repository](https://github.com/10xhub/10xScale Agentflow)** - Source code and issues
- **[PyPI Project](https://pypi.org/project/10xScale-Agentflow/)** - Package releases
- **[Examples Directory](https://github.com/10xhub/10xScale Agentflow/tree/main/examples)** - Runnable code samples

---

## 🙏 Contributing

Contributions are welcome! Please see our [GitHub repository](https://github.com/10xhub/10xScale Agentflow) for:

- Issue reporting and feature requests
- Pull request guidelines
- Development setup instructions
- Code style and testing requirements

---

## 💬 Support

- **Documentation**: [https://10xhub.github.io/10xScale Agentflow/](https://10xhub.github.io/10xScale Agentflow/)
- **Examples**: Check the [examples directory](https://github.com/10xhub/10xScale Agentflow/tree/main/examples)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/10xhub/10xScale Agentflow/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/10xhub/10xScale Agentflow/discussions)

---

**Ready to build intelligent agents?** Check out the [documentation](https://10xhub.github.io/10xScale Agentflow/) to get started!
