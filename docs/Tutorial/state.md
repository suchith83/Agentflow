# State & Messages: Managing Conversation Context

PyAgenity agents share a single contract for state and messaging so every node knows where to read/write context. This
tutorial explains the `AgentState` model, the `Message` primitives it stores, and how to extend both for your
application.

---

## 🧱 Why State Matters

- **Progress tracking** – `AgentState.execution_meta` keeps the current node, step count, interrupt flags, and error
	details so runs can pause and resume reliably.
- **Conversation context** – `AgentState.context` stores every `Message` exchanged between users, the assistant, and
	tools.
- **Persistence** – Because it is a Pydantic model, the state serialises cleanly for in-memory and database-backed
	checkpointers.

You rarely need a different interface: pass `state: AgentState` (or a subclass) to your nodes and tools and you can
query or mutate all of the execution data from one object.

---

## 🔍 Anatomy of `AgentState`

Defined in [`pyagenity/state/agent_state.py`](../../pyagenity/state/agent_state.py), the default state provides:

| Field | Type | Purpose |
|-------|------|---------|
| `context` | `list[Message]` | Ordered conversation history; new messages append here |
| `context_summary` | `str | None` | Optional condensed summary passed to models or memory stores |
| `execution_meta` | `ExecutionState` | Internal runtime metadata (current node, step, status flags) |

Convenience methods delegate to `execution_meta` so your nodes can call `state.advance_step()`,
`state.set_current_node(...)`, `state.set_interrupt(...)`, etc., without touching the underlying metadata object.

### Common helper methods

- `is_running()`, `is_interrupted()`, `is_stopped_requested()` – query execution status
- `complete()`, `error(msg)` – mark the run as finished or failed
- `clear_interrupt()` – resume after a pause

These helpers also emit logging so publishers (e.g. `ConsolePublisher`) can report transitions.

---

## 💬 Understanding `Message`

Messages are declared in [`pyagenity/utils/message.py`](../../pyagenity/utils/message.py). They support structured
content and tool calls, not just plain strings.

Key attributes:

- `role`: `"user" | "assistant" | "system" | "tool"`
- `content`: list of content blocks (text, tool results, media, etc.)
- `delta`: flag for streaming partials
- `tools_calls`: metadata generated when the assistant wants to invoke a tool
- `usages`: token accounting for providers that report it

Helpful constructors:

- `Message.text_message("hello")`
- `Message.tool_message([...])`
- `Message.from_response(...)` (used by `ModelResponseConverter`)

Utility method `message.text()` extracts best-effort human-readable text from mixed content blocks.

---

## 🛠️ Step-by-Step: Customising State

Let’s extend `AgentState` with application-specific fields and inspect how messages flow through the graph.

```python
# file: my_state.py
from pydantic import Field
from pyagenity.state.agent_state import AgentState
from pyagenity.utils import Message


class SupportState(AgentState):
		user_profile: dict = Field(default_factory=dict)
		pending_tasks: list[str] = Field(default_factory=list)


def create_initial_state() -> SupportState:
		state = SupportState()
		state.context.append(Message.text_message("Hello, I need help with my order."))
		state.user_profile = {"id": "cust-42", "tier": "gold"}
		return state
```

### Wire it into a graph

```python
from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import StateGraph
from pyagenity.utils import Message, ResponseGranularity


state = create_initial_state()
graph = StateGraph(state=state)


def echo(state: SupportState):
		reply = f"Hi {state.user_profile['id']}! How can I assist?"
		return [Message.text_message(reply, role="assistant")]


graph.add_node("MAIN", echo)
graph.add_edge("MAIN", "__end__")  # or use constants.START/END for clarity
graph.set_entry_point("MAIN")

app = graph.compile(checkpointer=InMemoryCheckpointer())

out = app.invoke({"messages": []}, config={"thread_id": "demo"}, response_granularity=ResponseGranularity.FULL)
print(out["state"]["user_profile"])  # {'id': 'cust-42', 'tier': 'gold'}
```

**Notice** how we returned a list of messages from the node. The invoke handler automatically appends them to
`state.context`, making them available to the next node.

---

## 🧪 Debugging Tips

- **Inspect full state** – pass `response_granularity=ResponseGranularity.FULL` when calling `invoke()` or `astream()`.
- **Log message flow** – use `for msg in state.context` inside nodes to print roles/content; combine with
	`ConsolePublisher` for richer traces.
- **Check interrupts** – before performing expensive work, call `state.is_stopped_requested()` to honour external stop
	requests.
- **Serialise your subclass** – run `state.model_dump()` (or `state.json()`) to ensure custom fields remain
	checkpointer-friendly.

---

## ✅ Checkpoints

| Goal | Verification |
|------|--------------|
| Custom fields persist | Run the graph twice with a database checkpointer and confirm the extra fields survive reload |
| Tool outputs captured | Ensure tool nodes return `Message.tool_message` so the result lands in `state.context` |
| Summaries stay small | Populate `context_summary` instead of keeping every message if you call LLMs frequently |
| No runtime refs | Audit state fields for non-serialisable handles (DB sessions, file descriptors, coroutines) |

---

## 📚 Where to Go Next

- Revisit the [Graph Fundamentals](graph.md) tutorial to see how state flows through a full agent
- Learn how to [wire tools and dependency injection](adapter.md) so messages can trigger real functions
- Explore [checkpointers and stores](checkpointer.md) to persist your customised state between runs

Once you’re comfortable with `AgentState` and `Message`, you can add richer tracking (user segments, retrieved documents,
metrics) without breaking the runtime contract.
# State Management in PyAgenity

State management is a crucial part of building agent graphs. The `AgentState` class in PyAgenity provides a small, consistent schema for storing conversational context, derived summaries, and internal execution metadata the runtime needs to track progress and interrupts.

This document explains what state represents in PyAgenity, why it's required, when you should extend it, and practical tips for using and subclassing `AgentState`.

## What is `AgentState`?

`AgentState` is a Pydantic model that acts as the canonical runtime state for an agent execution. It stores:

- `context`: a list of `Message` objects representing the message history or context items. The field is annotated to use the `add_messages` validator so messages are handled consistently when the state is created or updated.
- `context_summary`: an optional string holding a condensed summary of the context (useful for cost/latency optimizations when contacting LLMs or vector stores).
- `execution_meta`: an instance of `ExecutionState` (imported as `ExecMeta`) that contains internal-only metadata the runtime uses to track the current node, step counters, interrupts, errors, and completion status. The default initial value sets the `current_node` to the `START` constant.

The class also provides convenience methods that delegate to `execution_meta` for common runtime operations (interrupts, advancing steps, marking completion, etc.). These helpers keep agent code concise and avoid duplicated calls into the internal metadata object.

## Why have an explicit agent state?

There are a few strong reasons for a dedicated state model in agent systems:

- Persistence & Rehydration: Having a single serializable Pydantic model makes it trivial to persist an execution and later restore it (in-memory, database, or a checkpoint service).
- Observability: `execution_meta` centralizes runtime signals like "current node", step counters, and interrupts so publishers, logs, and debuggers can report consistent progress.
- Clear Contract: Node functions and tools can accept a single `state` parameter and know where to read/write context, summary, and engine metadata.
- Safer Subclassing: Applications commonly need custom fields (user profile, embeddings, or cached lookups). `AgentState` is designed to be subclassed while preserving the internal metadata and behavior.

## When should you use or extend `AgentState`?

Use `AgentState` as the state model for any agent graph that needs:

- Conversation context or tool outputs to be carried across nodes.
- Execution tracking for long-running flows (where interruptions, steps, or progress matter).
- Persistence between runs (e.g., to resume a paused or preempted execution).

Extend (subclass) `AgentState` when your application requires domain-specific fields. Examples:

- Storing a user's profile or session attributes.
- Adding a typed workspace for retrieved documents or embeddings.
- Adding counters or feature flags used by nodes.

Keep these guidelines when subclassing:

- Do not remove or rename the core fields (`context`, `context_summary`, `execution_meta`). The runtime expects them to exist.
- Add new fields with sensible defaults (use `Field(default_factory=...)` for mutable defaults).
- Avoid placing runtime-only constructs that can't be serialized (open DB connections, socket objects) on the state itself — persistence expects state to be JSON-serializable via Pydantic.

Example subclass:

```python
from pydantic import Field
from pyagenity.state.agent_state import AgentState

class MyAppState(AgentState):
		# an app-specific dictionary to cache lookups
		user_data: dict = Field(default_factory=dict)
		# a short flag used by nodes
		is_premium_user: bool = False

# usage: pass `MyAppState` when invoking/compiling graphs so injected states are typed
```

## AgentState API (convenience helpers)

The class provides several small helpers that delegate to `execution_meta`. These are intended to make node implementations and orchestration code more readable:

- `set_interrupt(node: str, reason: str, status, data: dict | None = None) -> None`
	- Mark an interrupt at `node` with a reason, a status code/object, and optional data.

- `clear_interrupt() -> None`
	- Clear any recorded interrupt.

- `is_running() -> bool` / `is_interrupted() -> bool`
	- Query execution status. Useful for nodes that need to short-circuit if the graph was cancelled.

- `advance_step() -> None`
	- Increment an internal step counter in `execution_meta`.

- `set_current_node(node: str) -> None`
	- Update the `current_node` value tracked in the metadata.

- `complete() -> None` / `error(error_msg: str) -> None`
	- Mark the execution as finished or errored.

- `is_stopped_requested() -> bool`
	- Check whether a stop was requested; nodes can use this to gracefully bail out.

These helpers log at appropriate levels (debug/info/error) to help with tracing.

## Messages and `context`

The `context` field stores a list of `Message` objects (see `pyagenity.utils.message.Message`). When you add messages to state, prefer the helper functions and factories available in `Message` (for example `Message.from_text(...)`, `Message.tool_message(...)`, or `Message.from_response(...)`).

The `add_messages` validator/annotation ensures messages are normalized when the state is constructed so code reading `state.context` can rely on a consistent shape.

## Persistence and Checkpointer considerations

`AgentState` is intentionally serializable (Pydantic model), which makes it compatible with the checkpointers used by the graph (in-memory, Postgres, Redis backed, etc.). When you persist state:

- Keep the state small: prefer summarizing long conversations into `context_summary` rather than keeping the full message history indefinitely.
- Avoid placing large binary blobs or non-serializable objects on the state.

If you use `PgCheckpointer` or other external checkpointers, ensure your custom fields are simple Python types or Pydantic-friendly models.

## Practical usage patterns

- Short-running stateless nodes: nodes may accept `state` as input but only use `context_summary` or transient inputs — still pass `AgentState` so the runtime has a canonical place to read/write progress.
- Long-running/resumable flows: rely on `execution_meta` to persist the `current_node`, step, and interrupts. When resuming, set the right `current_node` before invoking the compiled graph.
- Tool integrations: tools that perform external calls can write tool results into `state.context` as tool messages — this allows later nodes (or the caller) to inspect tool outputs.

## Tips and gotchas

- Always reference `START` and `END` constants when setting entry points or routing — `AgentState`'s default `execution_meta` current node is set to `START` so your graphs start in a consistent state.
- Keep the state serializable. When in doubt, try `state.json()` to confirm Pydantic serialization works for your subclass fields.
- Don't store ephemeral runtime handles on the state (DB connections, asyncio tasks). Instead, use dependency injection for those resources.
- Use `context_summary` to reduce cost when calling LLMs or creating embeddings — you can keep the full `context` for audit logs while passing only the summary to external models.

## Small example (flow)

1. Create a `MyAppState` instance with initial user info.
2. Build a `StateGraph` and set the entry node.
3. Add nodes that accept `(state: MyAppState, ...)` and update `state.context` with messages using `Message.from_text(...)`.
4. Persist intermediate state via the graph's checkpointer if you need to resume later.

## Summary

`AgentState` is the lightweight, serializable backbone of PyAgenity agent executions. It bundles conversation context, a summary for efficient model calls, and internal execution metadata used by the runtime to track and resume work. Subclass it for app-specific fields, keep it serializable, and use the provided helpers to manage interrupts, progress, and completion.

If you'd like, I can add a small runnable example that demonstrates creating a subclass, wiring it into a minimal `StateGraph`, and showing persistence with the in-memory checkpointer.