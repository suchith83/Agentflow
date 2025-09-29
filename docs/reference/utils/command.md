## Command API

The `Command` object lets a node return both a state (or message) update and an instruction about where the graph should go next. This mirrors the pattern found in LangGraph, giving you explicit control over control-flow without mutating internal graph structures mid-run.

Import path: `from pyagenity.utils.command import Command`

### When to Use

Use `Command` when a node needs to:

- Route dynamically to a different node (conditional branching not known at compile time)
- Terminate early by jumping to `END`
- Bubble execution to a parent graph (future nested graph scenarios) using `Command.PARENT`
- Return a `Message` or partial state update plus a jump target in a single return value

If your branching is static and determined purely by inspecting the current state, prefer `add_conditional_edges()` at graph construction—it's simpler and validated up front.

### Constructor Signature

```python
Command(
	update: AgentState | Message | str | BaseConverter | None = None,
	goto: str | None = None,
	graph: str | None = None,
	state: AgentState | None = None,
)
```

Field meanings:

| Field | Meaning |
|-------|---------|
| `update` | A state delta, a single `Message`, a plain string (wrapped as message), or a `BaseConverter` (e.g. `ModelResponseConverter`) to be normalised into messages. |
| `goto` | The name of the next node to execute (or `END`). |
| `graph` | Reserved for multi-graph navigation. Use `Command.PARENT` to return to parent when nested graphs are supported. |
| `state` | Explicit state object to attach (rare—normally you just return `update`). |

### Basic Example

```python
from pyagenity.graph import StateGraph
from pyagenity.utils import Message, END
from pyagenity.utils.command import Command


def router(state, config):
	text = state.context[-1].text() if state.context else ""
	if "weather" in text:
		return Command(goto="WEATHER")
	if "bye" in text:
		return Command(update=Message.text_message("Goodbye!"), goto=END)
	return Command(update=Message.text_message("I can help with weather queries."), goto="MAIN")


def main(state, config):
	return [Message.text_message("Ask me about the weather or say bye.")]


graph = StateGraph()
graph.add_node("ROUTER", router)
graph.add_node("MAIN", main)
graph.add_node("WEATHER", lambda s, c: [Message.text_message("It's sunny.")])
graph.add_edge("ROUTER", "MAIN")  # fallback if router returns nothing
graph.add_edge("MAIN", END)
graph.add_edge("WEATHER", END)
graph.set_entry_point("ROUTER")
app = graph.compile()
```

### Using with Model Responses

If a node returns a `ModelResponseConverter`, you can still wrap it in a `Command` to jump afterwards:

```python
def llm_step(state, config):
	converter = ModelResponseConverter(response=call_llm(...), converter="litellm")
	return Command(update=converter, goto="POST_PROCESS")
```

### Best Practices

- Keep `goto` targets explicit; avoid dynamically constructing names that aren't declared—validation can't help you then.
- Prefer simple returns (list[Message] or state) when no control jump is required; reserve `Command` for clarity around routing.
- Combine with `add_conditional_edges` for coarse routing, and `Command` for fine-grained per-node decisions.

### Error Handling

If `goto` references a node that does not exist or creates a cycle exceeding the recursion limit, execution will raise during traversal (handled by the invoke handler). Always test new control paths with a small `recursion_limit` in config.

### Reference

::: pyagenity.utils.command

---

For advanced graph flow patterns (interrupts, nested graphs, background tasks), see forthcoming sections in the control-flow tutorial.
