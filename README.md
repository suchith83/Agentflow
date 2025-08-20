
# PyAgenity

![PyPI](https://img.shields.io/pypi/v/pyagenity?color=blue)
![License](https://img.shields.io/github/license/Iamsdt/pyagenity)
![Python](https://img.shields.io/pypi/pyversions/pyagenity)

**PyAgenity** is a lightweight Python framework for building intelligent agents and orchestrating multi-agent workflows on top of the [LiteLLM](https://github.com/BerriAI/litellm) unified LLM interface.

PyAgenity is a lightweight Python framework for building intelligent agents and orchestrating multi-agent workflows on top of the LiteLLM unified LLM interface.


---

## Features


- Unified `Agent` abstraction (no raw LiteLLM objects leaked)
- Structured responses with `content`, optional `thinking`, and `usage`
- Streaming support with incremental chunks
- Final message hooks for persistence/logging
- LangGraph-inspired graph engine: nodes, conditional edges, pause/resume (human-in-loop)
- In-memory session state store (pluggable in the future)


---

## Installation

**With [uv](https://github.com/astral-sh/uv) (recommended):**

```bash
uv pip install pyagenity
```

Or with pip:

```bash
pip install pyagenity
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

## Example: React Weather Agent

This repository includes a small example agent that demonstrates tool injection and a simple tool node for returning weather information. The example is in `examples/react/react_weather_agent.py` and uses an in-memory checkpointer. It's intended as a runnable demo showing how to register ToolNodes, conditional edges, and invoke the compiled graph.

Key points:
- Demonstrates an injectable tool signature that receives `tool_call_id` and `state`.
- Shows how to add `ToolNode` and conditional edges into a `StateGraph`.
- Uses `litellm.completion` to call an LLM (set your provider keys as environment vars or use a `.env`).

Excerpt (simplified) from `examples/react/react_weather_agent.py`:

```python
from dotenv import load_dotenv
from litellm import completion

from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import StateGraph, ToolNode
from pyagenity.state.agent_state import AgentState
from pyagenity.utils import Message
from pyagenity.utils.constants import END
from pyagenity.utils.converter import convert_messages
from pyagenity.utils.injectable import InjectState, InjectToolCallID

load_dotenv()

checkpointer = InMemoryCheckpointer()

def get_weather(location: str, tool_call_id: InjectToolCallID, state: InjectState) -> str:
    """Simple demo tool that prints injected params and returns a fake weather string."""
    if tool_call_id:
        print(f"Tool call ID: {tool_call_id}")
    if state and hasattr(state, "context"):
        print(f"Number of messages in context: {len(state.context)}")  # type: ignore
    return f"The weather in {location} is sunny."

# Build graph and compile
tool_node = ToolNode([get_weather])

def main_agent(state: AgentState, config: dict[str, any], checkpointer=None, store=None):
    # Build messages and call the LLM. Use tools when appropriate.
    prompts = """
        You are a helpful assistant.
        Your task is to assist the user in finding information and answering questions.
    """

    messages = convert_messages(system_prompts=[{"role": "system", "content": prompts}], state=state)

    # If last message is a tool result, return final response without tools
    if state.context and state.context[-1].role == "tool" and state.context[-1].tool_call_id is not None:
        response = completion(model="your-model", messages=messages)
    else:
        response = completion(model="your-model", messages=messages, tools=tool_node.all_tools())

    return response

# Build graph, compile and invoke (see full file for details)
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
- The example uses `litellm`'s `completion` function — set `model` to a provider/model available in your environment (for example `gpt-4o-mini` or other supported model strings).
- `InMemoryCheckpointer` is for demo/testing only. Replace with a persistent checkpointer for production.



---

## Human-in-the-Loop


A `HumanInputNode` causes execution to pause (`WAITING_HUMAN`) if `human_input` is absent. Provide input via `resume(session_id, human_input=...)`.


---

## Final Hooks


Use `GraphExecutor.add_final_hook(callable)` to register hooks invoked when a session reaches `COMPLETED` or `FAILED`.


---

## Roadmap


- Persistent state backend (Redis, SQL, etc.)
- Parallel / branching strategies and selection policies
- Tool invocation nodes & function calling wrappers
- Tracing / telemetry integration


---

## License

MIT

---

## Project Links

- [GitHub Repository](https://github.com/Iamsdt/pyagenity)
- [PyPI Project Page](https://pypi.org/project/pyagenity/)

---

## Publishing to PyPI

1. **Build the package:**
    ```bash
    uv pip install build twine
    python -m build
    ```
2. **Upload to PyPI:**
    ```bash
    uv pip install twine
    twine upload dist/*
    ```

For test uploads, use [TestPyPI](https://test.pypi.org/):
```bash
twine upload --repository testpypi dist/*
```
