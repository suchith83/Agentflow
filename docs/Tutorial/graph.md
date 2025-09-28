# Graph Fundamentals: Build Your First PyAgenity Agent

Learn how to assemble a runnable agent graph using `StateGraph`, nodes, and edges. We will reproduce the flow in
`examples/react/react_sync.py`, explain each piece, and give you checkpoints to validate that everything works.

---

## 📦 What You Will Build

- A minimal agent graph with:
  - `StateGraph` orchestrating execution
  - A main node that talks to an LLM (via LiteLLM)
  - A `ToolNode` that exposes your own Python function
- A compiled application you can run synchronously with `CompiledGraph.invoke()`

Refer to the completed example in [`examples/react/react_sync.py`](../../examples/react/react_sync.py) while you follow along.

---

## ✅ Prerequisites

- Python 3.12+
- An LLM provider supported by [LiteLLM](https://docs.litellm.ai/docs/) (OpenAI, Gemini, etc.)
- Environment variables set in `.env` (e.g. `OPENAI_API_KEY` or `GEMINI_API_KEY`)
- Optional but recommended: create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyagenity[litellm]
pip install python-dotenv
```

---

## 🛠️ Step 1 – Create the Project Skeleton

1. Clone the repo or copy the `examples/react/` directory into your workspace.
2. Open `react_sync.py`. We will build a simplified version from scratch below so you understand every line.

---

## 🧩 Step 2 – Write a Tool Function

Tools let your graph call regular Python functions. PyAgenity will inject useful context into the function at runtime.

```python
from pyagenity.state.agent_state import AgentState


def get_weather(
	location: str,
	tool_call_id: str | None = None,
	state: AgentState | None = None,
) -> str:
	"""Return mock weather information for the demo."""
	if tool_call_id:
		print(f"Tool call ID: {tool_call_id}")
	if state:
		print(f"Context length: {len(state.context)} messages")

	return f"The weather in {location} is sunny"
```

Why it matters:

- Arguments like `tool_call_id` and `state` are **auto-injected** thanks to InjectQ
- Functions can return strings, messages, or `Message.tool_message` payloads

Wrap your function in a `ToolNode` so the graph knows how to call it:

```python
from pyagenity.graph import ToolNode


tool_node = ToolNode([get_weather])
```

---

## 🧠 Step 3 – Implement the Main Agent Node

Your main node usually calls the LLM. We will use LiteLLM’s `completion()` helper and PyAgenity’s response converter to keep things framework-agnostic.

```python
from litellm import completion
from pyagenity.adapters.llm.model_response_converter import ModelResponseConverter
from pyagenity.utils.converter import convert_messages


def main_agent(state: AgentState):
	system_prompt = """
		You are a helpful assistant. Call tools when the user asks for factual data.
	"""

	messages = convert_messages(
		system_prompts=[{"role": "system", "content": system_prompt}],
		state=state,
	)

	# Call all registered tools when asking the model to reason
	tools = tool_node.all_tools_sync()
	response = completion(
		model="gemini/gemini-2.5-flash",
		messages=messages,
		tools=tools,
	)

	return ModelResponseConverter(response, converter="litellm")
```

Key takeaways:

- `convert_messages()` merges the existing state context with prompts in LiteLLM format
- `ModelResponseConverter` normalises provider responses back into PyAgenity `Message` objects

---

## 🧭 Step 4 – Assemble the Graph

Now wire the nodes into a `StateGraph` and define how execution flows. Always include `START` and `END` sentinels.

```python
from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import StateGraph
from pyagenity.utils.constants import END


checkpointer = InMemoryCheckpointer()


def should_use_tools(state: AgentState) -> str:
	# If the assistant requested a tool, go to TOOL, otherwise finish
	if state.context and getattr(state.context[-1], "tools_calls", None):
		return "TOOL"
	return END


graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)

# MAIN -> conditional routing
graph.add_conditional_edges(
	"MAIN",
	should_use_tools,
	{"TOOL": "TOOL", END: END},
)

# TOOL -> MAIN (loop back for final response)
graph.add_edge("TOOL", "MAIN")

# Entry point is implicit once you add START -> MAIN
graph.set_entry_point("MAIN")

app = graph.compile(checkpointer=checkpointer)
```

What to notice:

- `add_node()` accepts either a callable or a pre-created `ToolNode`
- `add_conditional_edges()` maps return values from the condition to downstream nodes
- `set_entry_point()` inserts the `START` edge for you
- Compiling binds services (checkpointer, publishers, DI container) and yields a `CompiledGraph`

---

## ▶️ Step 5 – Run the Agent

Invoke the compiled graph with an initial message payload. Every run needs a `thread_id` to separate conversations.

```python
from pyagenity.utils import Message


if __name__ == "__main__":
	input_data = {
		"messages": [
			Message.text_message(
				"Please call the get_weather function for New York City"
			)
		]
	}
	config = {"thread_id": "demo-thread", "recursion_limit": 10}

	result = app.invoke(input_data, config=config)

	for msg in result["messages"]:
		print("---", msg.role.upper(), "---")
		print(msg.text())
```

Expected output (truncated):

```
Tool call ID: weather-tool-123
Context length: 1 messages
--- TOOL ---
The weather in New York City is sunny
--- ASSISTANT ---
Here is the weather...
```

If you see a `GraphError: No entry point set`, double-check that you called `set_entry_point()` before `compile()`.

---

## 🔍 Checkpoints & Troubleshooting

| What to verify | How |
|----------------|-----|
| Dependency injection works | Add print statements in `get_weather` to inspect `tool_call_id` and `state` |
| Messages accumulate | Inspect `state.context` after `invoke()` or set `response_granularity=ResponseGranularity.FULL` |
| Graph reaches END | Ensure `should_use_tools` returns `END` when no tool call is pending |
| Recursion guard | Adjust `config["recursion_limit"]` to prevent runaway loops |

Common issues:

- **Missing API key** – LiteLLM will raise an authentication error. Confirm the `.env` file is loaded via `python-dotenv`.
- **Tools never fire** – Ensure the assistant message sets `tools_calls` in the response. Some models require explicit prompting.

---

## 📚 Next Steps

You now have a working graph and know how nodes execute. Proceed to:

- **[State & Messages](state.md)** to explore richer state models
- **[Tools & Dependency Injection](adapter.md)** for more advanced tool wiring and MCP integration
- **[Control Flow & Routing](graph.md#control-flow)** (later in this guide) to learn about interrupts and branching patterns

Keep iterating on your `StateGraph` by adding more nodes, routing logic, or plugging in publishers and stores.
