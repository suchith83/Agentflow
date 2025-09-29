# Graph Architecture

PyAgenity's graph system orchestrates agent reasoning, tool execution, and stateful message flow. It is intentionally
minimal, composable, and DI-friendly. This section is your conceptual map.

---

## Core Building Blocks

| Concept | File(s) | Role |
|---------|---------|------|
| `StateGraph` | `pyagenity/graph/state_graph.py` | Declarative builder: register nodes, edges, tools, conditions |
| `Node` | `pyagenity/graph/node.py` | Wrapper around user function (sync/async) with DI injection |
| `ToolNode` | `pyagenity/graph/tool_node/` | Tool registry + dispatcher (local + external providers) |
| `Edge` | `pyagenity/graph/edge.py` | Directional link; supports conditional routing |
| `CompiledGraph` | `pyagenity/graph/compiled_graph.py` | Runtime engine: invoke, stream, checkpoint, publish |
| `Command` | `pyagenity/utils/command.py` | Inline control object for dynamic goto / updates |

Supporting utilities: converters (LLM output → `Message` blocks), id + thread generators, background task manager,
callback + publisher subsystems.

---

## Lifecycle at a Glance

1. Build: create `StateGraph`, add nodes & edges
2. Configure: set entry point, optionally conditional edges, tool nodes
3. Compile: dependency container frozen, checkpointer/publisher bound, internal handlers instantiated
4. Execute: `invoke()` (batch) or `stream()/astream()` (incremental)
5. Persist / Publish: events & state snapshots emitted, usage + tool calls recorded
6. Resume / Stop: interrupted runs can continue; stop flags honored mid-execution

---

## Node Function Contract

A node function usually:

```python
async def my_node(state: AgentState, config: dict, ...injected_deps) -> State | list[Message] | Command | ModelResponseConverter:
    ...logic...
    return [...]
```

Return types supported:

- `list[Message]` – append messages to context
- `AgentState` (or subclass) – full state replacement/merge
- `Command` – state/message update + jump target
- `ModelResponseConverter` – deferred LLM response normalisation
- `None` – treated as no-op

Tool nodes share the same return semantics but are triggered by tool call entries in an assistant message.

---

## Design Principles

- **Explicit over magic** – You wire nodes and edges; runtime doesn't infer hidden transitions.
- **State-first** – All decisions emerge from `AgentState`. Determinism improves testing & resuming.
- **Pluggable IO** – Checkpointers, stores, publishers, ID generators, tools all injectable.
- **Provider-agnostic** – LLM specifics isolated in converters; tool registries abstract away host APIs.
- **Composable** – Future nested graphs and higher-order agents (routers, supervisors) build on same primitives.

---

## Quick Mental Model

Think of the graph as an event loop over a mutable state object:

```
while not done:
  node = current_node()
  output = run(node, state)
  apply(output)
  advance()
```

Where `apply()` merges messages, updates metadata, handles `Command` jumps, and triggers tool execution when tool call
blocks are present.

---

## Where to Next?

Dive deeper:

- Nodes & Return Types (`nodes.md`)
- Control Flow & Edges (`control_flow.md`)
- Tools: Local, MCP, Composio, LangChain (`tools.md`)
- Execution & Streaming Runtime (`execution.md`)
- Advanced Patterns & Performance (`advanced.md`)

Or jump back to higher-level tutorials once concepts click.
