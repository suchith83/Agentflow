# Agentflow — Graph-Level Lifecycle Hooks: Implementation Plan

---

## Context: What Exists Today

The `CallbackManager` in `agentflow/utils/callbacks.py` already provides **invocation-level** hooks that fire inside a single node's AI/Tool/MCP call:

| Existing Hook | Level | Fires |
|---|---|---|
| `before_invoke` | Per-invocation | Before each AI / Tool / MCP call inside a node |
| `after_invoke` | Per-invocation | After each AI / Tool / MCP call inside a node |
| `on_error` | Per-invocation | When any AI / Tool / MCP call throws |
| `execute_validators` | Per-invocation | On incoming messages before they enter a node |

These live deep in the call stack: `Node.execute()` → `InvokeNodeHandler` / `StreamNodeHandler` → the actual LLM/tool adapter. They are **not** aware of the graph's overall lifecycle — when it starts, ends, checkpoints, or pauses.

The 7 new hooks operate at the **graph orchestration level**: `InvokeHandler._execute_graph()`, `StreamHandler._execute_graph()`, and the shared utility `sync_data()`.

---

## Execution Flow (Annotated)

Understanding exactly where each hook slots in requires reading the full execution pipeline. Here is the annotated flow for both `invoke` and `stream` paths:

```
CompiledGraph.ainvoke() / astream()
  └─ _prepare_config()                          ← generates thread_id, run_id
  └─ InvokeHandler.invoke() / StreamHandler.stream()
       ├─ load_or_create_state()                ← load from checkpointer or create fresh
       ├─ check_interrupted()                   ← detect resume vs fresh start
       │     └─ if state.is_interrupted():
       │                                        ◀── [on_resume fires HERE]
       │           config["resume_data"] = input_data
       │           state.clear_interrupt()
       ├─ ── ── ── ── ── ── ── ── ── ── ──     ◀── [on_graph_start fires HERE]
       └─ _execute_graph(state, config)
            │
            ├─ while current_node != END and step < max_steps:
            │     ├─ check_stop_requested()
            │     ├─ state.set_current_node()
            │     ├─ call_realtime_sync()
            │     ├─ check_and_handle_interrupt("before")
            │     │                              ◀── [on_interrupt fires before sync_data]
            │     ├─ node.execute() / node.stream()
            │     ├─ process result → update state
            │     │                              ◀── [on_state_update fires HERE]
            │     ├─ call_realtime_sync()
            │     ├─ check_and_handle_interrupt("after")
            │     │                              ◀── [on_interrupt fires before sync_data]
            │     ├─ get_next_node()
            │     └─ step += 1, state.advance_step()
            │
            ├─ state.complete()
            │                                   ◀── [on_graph_end fires HERE]
            ├─ sync_data(trim=True)
            │                                   ◀── [on_checkpoint fires before persistence]
            │     └─ checkpointer.aput_state()
            │
            └─ except Exception:
                  state.error()
                                                ◀── [on_graph_error fires HERE]
                  sync_data()
                  raise
```

---

## New Types to Add in `callbacks.py`

### `GraphLifecycleContext` (shared context dataclass)

All 7 hooks receive this as their first argument. It gives the hook enough metadata to identify which execution is running.

```python
@dataclass
class GraphLifecycleContext:
    config: dict[str, Any]  # full config dict passed to invoke/stream
    timestamp: str          # config["timestamp"] — ISO8601 start time
    metadata: dict[str, Any] | None = None  # open-ended extra context

    @property
    def thread_id(self) -> str | None:
        return self.config.get("thread_id")

    @property
    def run_id(self) -> str | None:
        return self.config.get("run_id")
```

**Why not reuse `CallbackContext`?** `CallbackContext` carries `invocation_type` and `function_name` which are node-invocation concepts. Graph lifecycle hooks don't have an invocation type — they're structural events of the entire graph run.

---

### `GraphLifecycleHook` (abstract base class)

All methods have **default no-op implementations** so users only override what they need. This is the primary API surface.

```python
class GraphLifecycleHook(ABC):
    """
    Base class for graph-level lifecycle hooks.
    
    Override only the methods you need. All methods are async.
    Hooks that can modify execution return replacement data. Return None to keep current data.
    """

    async def on_graph_start(
        self,
        context: GraphLifecycleContext,
        state: AgentState,
    ) -> AgentState | None:
        """Called after state is loaded but before the first node executes.
        Return a modified AgentState to replace the initial state, or None to keep original."""
        return None

    async def on_graph_end(
        self,
        context: GraphLifecycleContext,
        final_state: AgentState,
        messages: list[Message],
        total_steps: int,
    ) -> AgentState | None:
        """Called after execution loop completes, before final persistence.
        Return a modified AgentState to replace the final state, or None to keep original."""
        return None

    async def on_graph_error(
        self,
        context: GraphLifecycleContext,
        error: Exception,
        partial_state: AgentState,
        messages: list[Message],
        step: int,
        node_name: str,
    ) -> tuple[AgentState, str] | None:
        """Called when an unhandled exception escapes the graph loop.
        Return a modified (AgentState, error_message) tuple for the persisted error snapshot,
        or None to keep original.
        Cannot suppress the error — always re-raised after this hook."""
        return None

    async def on_interrupt(
        self,
        context: GraphLifecycleContext,
        interrupted_node: str,
        interrupt_type: str,   # "before" | "after" | "stop" | "remote_tool"
        state: AgentState,
    ) -> AgentState | None:
        """Called when graph execution pauses, before interrupt state is persisted.
        Return a modified AgentState to persist and return to the caller, or None to keep original."""
        return None

    async def on_resume(
        self,
        context: GraphLifecycleContext,
        resumed_node: str,
        state: AgentState,
        resume_data: dict[str, Any],
    ) -> AgentState | None:
        """Called when a previously interrupted graph is resumed, before clear_interrupt().
        Return a modified AgentState to continue execution, or None to keep original.
        resume_data is mutable in-place and is copied into config["resume_data"] after this hook."""
        return None

    async def on_checkpoint(
        self,
        context: GraphLifecycleContext,
        state: AgentState,
        messages: list[Message],
        is_context_trimmed: bool,
    ) -> tuple[AgentState, list[Message]] | AgentState | None:
        """Called immediately before state/messages are persisted.
        Return a modified AgentState, or (AgentState, messages), or None to keep current data."""
        return None

    async def on_state_update(
        self,
        context: GraphLifecycleContext,
        node_name: str,
        old_state: AgentState,
        new_state: AgentState,
        step: int,
    ) -> AgentState | None:
        """Called after each node produces a result and state is updated.
        Return a modified AgentState to override the new state, or None to keep it."""
        return None
```

**Functional callable type aliases** (for users who prefer lambdas or plain `async def` functions):

```python
OnGraphStartCallbackType = Union[
    GraphLifecycleHook,
    Callable[[GraphLifecycleContext, AgentState], Awaitable[AgentState | None]],
]
OnGraphEndCallbackType = Callable[
    [GraphLifecycleContext, AgentState, list[Message], int],
    Awaitable[AgentState | None],
]
OnGraphErrorCallbackType = Callable[
    [GraphLifecycleContext, Exception, AgentState, list[Message], int, str],
    Awaitable[tuple[AgentState, str] | None],
]
OnInterruptCallbackType = Callable[
    [GraphLifecycleContext, str, str, AgentState],
    Awaitable[AgentState | None],
]
OnResumeCallbackType = Callable[
    [GraphLifecycleContext, str, AgentState, dict[str, Any]],
    Awaitable[AgentState | None],
]
OnCheckpointCallbackType = Callable[
    [GraphLifecycleContext, AgentState, list[Message], bool],
    Awaitable[tuple[AgentState, list[Message]] | AgentState | None],
]
OnStateUpdateCallbackType = Callable[
    [GraphLifecycleContext, str, AgentState, AgentState, int],
    Awaitable[AgentState | None],
]
```

---

### Mutability Rule

Lifecycle hooks should be mutable only when the runtime can still honor their returned data:

- If the hook fires before persistence, return values are applied before `sync_data()` writes to the checkpointer.
- If the hook fires before returning to the caller, return values are applied to the state that `ainvoke()` / `astream()` exposes.
- If the hook fires on error, return values only change the persisted error snapshot; they do not recover or suppress the exception.
- Hook implementations may mutate the passed `AgentState` in place, but returning the replacement state is the preferred explicit contract.

This is why the previous read-only design was too conservative for `on_interrupt`, `on_resume`, `on_checkpoint`, and `on_graph_end`: the current code has clear pre-persistence points where those mutations can be safely applied. The only remaining hard boundary is error suppression, which should stay unsupported at graph level to avoid hiding failed production runs.

---

### `CallbackManager` additions

```python
class CallbackManager:
    def __init__(self):
        # ... existing registries ...
        
        # New: graph lifecycle hooks (ordered list — executed in registration order)
        self._lifecycle_hooks: list[GraphLifecycleHook] = []

    # ── Registration ────────────────────────────────────────────────

    def register_lifecycle_hook(self, hook: GraphLifecycleHook) -> None:
        """Register a full lifecycle hook (object implementing GraphLifecycleHook)."""
        self._lifecycle_hooks.append(hook)

    # ── Execution helpers (called by InvokeHandler / StreamHandler) ─

    async def fire_on_graph_start(self, context, state) -> AgentState: ...
    async def fire_on_graph_end(self, context, final_state, messages, total_steps) -> AgentState: ...
    async def fire_on_graph_error(
        self, context, error, partial_state, messages, step, node
    ) -> tuple[AgentState, str] | None: ...
    async def fire_on_interrupt(self, context, interrupted_node, interrupt_type, state) -> AgentState: ...
    async def fire_on_resume(self, context, resumed_node, state, resume_data) -> AgentState: ...
    async def fire_on_checkpoint(self, context, state, messages, is_trimmed) -> tuple[AgentState, list[Message]]: ...
    async def fire_on_state_update(self, context, node_name, old_state, new_state, step) -> AgentState: ...
```

---

## Per-Hook Specification

---

### 1. `on_graph_start`

**Purpose**: Let users modify the initial state or perform setup before any node runs.

**Input**:
| Parameter | Type | Source |
|---|---|---|
| `context` | `GraphLifecycleContext` | Built from `config` in `invoke()` / `stream()` |
| `state` | `AgentState` | Result of `load_or_create_state()` — already has messages merged |

**Output**: `AgentState | None`
- Return `None` → use the original state unchanged
- Return a modified `AgentState` → this replaces the initial state going into `_execute_graph()`

**Fire location**: `InvokeHandler.invoke()` and `StreamHandler.stream()`, after `check_interrupted()`, before calling `_execute_graph()`

**Code point** (`invoke_handler.py`):
```python
# After: config = await check_interrupted(state, input_data, config)
# Before: final_state, messages = await self._execute_graph(state, config)

state = await callback_mgr.fire_on_graph_start(
    GraphLifecycleContext(
        config=config,
        timestamp=config["timestamp"],
    ),
    state,
) or state
```

**Use cases**:
- Inject observability trace IDs into `state.execution_meta.internal_data`
- Pre-populate custom state fields from external sources (e.g., user profile DB lookup)
- Set rate-limiting counters or request budgets on state
- Initialize OpenTelemetry spans and store span context in state
- Add audit metadata (who triggered this run, from which service)
- Modify config-driven state fields (e.g., set language preference)

**What can go wrong without it**:
- No way to enrich state before execution — users have to pre-process before calling `ainvoke()`
- Observability tooling has no clean hook to start a trace span

---

### 2. `on_graph_end`

**Purpose**: Perform cleanup, finalization, notifications, or metric recording after successful execution.

**Input**:
| Parameter | Type | Source |
|---|---|---|
| `context` | `GraphLifecycleContext` | Same as `on_graph_start` |
| `final_state` | `AgentState` | State after `state.complete()`, before final `sync_data()` |
| `messages` | `list[Message]` | All messages produced during the graph run |
| `total_steps` | `int` | Final step count from `step` variable in the loop |

**Output**: `AgentState | None`
- Return `None` → persist and return the completed state unchanged
- Return a modified `AgentState` → persist and return that final state

**Fire location**: `InvokeHandler._execute_graph()` after `state.complete()`, before `sync_data(trim=True)`. Same in `StreamHandler._execute_graph()` before final persistence and final yield.

**Code point** (`invoke_handler.py`, inside `_execute_graph`):
```python
# After: state.complete()
# Before: await sync_data(...); return state, messages

state = await callback_mgr.fire_on_graph_end(
    context, final_state=state, messages=messages, total_steps=step
)
await sync_data(state=state, config=config, messages=messages, trim=True)
return state, messages
```

**Use cases**:
- Apply final state normalization before persistence (e.g., final summary or compact metadata)
- Add completion audit metadata directly to `state.execution_meta.internal_data`
- Record execution duration, step count, message count to metrics (Prometheus, Datadog)
- Send Slack/email notification when a long-running agent completes
- Close OpenTelemetry spans with success status
- Persist final summary to an external analytics database
- Trigger downstream workflows (webhooks, SQS messages)
- Archive conversation transcript to cold storage

**What can go wrong without it**:
- Post-execution business logic (e.g., "send email when research is complete") must be in application code outside the library, coupling business logic to calling code

---

### 3. `on_graph_error`

**Purpose**: Capture, log, and alert on unhandled failures that escape the execution loop.

**Input**:
| Parameter | Type | Source |
|---|---|---|
| `context` | `GraphLifecycleContext` | Same context |
| `error` | `Exception` | The exception from the `except` block |
| `partial_state` | `AgentState` | State at time of error (after `state.error()`) |
| `messages` | `list[Message]` | Messages collected before the error |
| `step` | `int` | Step number where error occurred |
| `node_name` | `str` | `current_node` at time of error |

**Output**: `tuple[AgentState, str] | None`
- Return `None` → persist the current error state unchanged and keep the raised exception message
- Return a modified `(AgentState, error_message)` tuple → persist that modified error snapshot and use the returned message
- The hook **cannot suppress the error**. The exception is always re-raised after this hook.

> **Note**: Recovery from node-level errors is handled by the existing `on_error` callback on `CallbackManager` (invocation level). `on_graph_error` fires only for errors that escape the entire loop.

**Fire location**: `InvokeHandler._execute_graph()` and `StreamHandler._execute_graph()`, in the `except Exception` block, after `state.error()`, before `sync_data()` and `raise`.

**Code point** (`invoke_handler.py`):
```python
except Exception as e:
    logger.exception("Graph execution failed: %s", e)
    state.error(str(e))
    event.event_type = EventType.ERROR
    publish_event(event)

    # NEW:
    result = await callback_mgr.fire_on_graph_error(
        context, error=e, partial_state=state,
        messages=messages, step=step, node_name=current_node
    )
    if result is not None:
        state, error_message = result
    await sync_data(state=state, config=config, messages=messages, trim=True)
    raise
```

**Use cases**:
- Attach structured failure diagnostics to `state.execution_meta.internal_data`
- Mask or compact sensitive partial state before persisting the failed checkpoint
- Alert PagerDuty / OpsGenie on production failures
- Log structured error data to Sentry / Datadog APM
- Capture conversation state at failure point for post-mortem debugging
- Close OpenTelemetry spans with error status and exception attributes
- Record failure metrics by node name (identify which node breaks most often)
- Send failure webhook to client system

**What can go wrong without it**:
- Production errors in agent graphs are silent unless application code wraps every `ainvoke()` in try/except — fragile and duplicates error handling logic

---

### 4. `on_interrupt`

**Purpose**: Notify external systems (UI, webhooks, databases) when execution pauses and waits for input.

**Input**:
| Parameter | Type | Source |
|---|---|---|
| `context` | `GraphLifecycleContext` | Same context |
| `interrupted_node` | `str` | The node where execution paused |
| `interrupt_type` | `str` | `"before"` \| `"after"` \| `"stop"` \| `"remote_tool"` |
| `state` | `AgentState` | State at interrupt point (already has `interrupted_node` set in `execution_meta`) |

**Output**: `AgentState | None`
- Return `None` → persist and return the interrupt state unchanged
- Return a modified `AgentState` → persist and return that interrupt state

**Fire locations**:
1. In `check_and_handle_interrupt()` — after `state.set_interrupt(...)`, before `sync_data(...)` (covers `"before"` and `"after"` types)
2. In `interrupt_graph()` — after `state.set_interrupt(...)`, before `sync_data(...)` (covers `"remote_tool"` type)
3. In `check_stop_requested()` — after `state.set_interrupt(...)`, before `sync_data(...)` (covers `"stop"` type)

**Code point** (`heandler_utils.py`, inside interrupt utility):
```python
state.set_interrupt(current_node, f"interrupt_{interrupt_type}: {current_node}", status)
state = await callback_mgr.fire_on_interrupt(
    context, interrupted_node=current_node, interrupt_type=interrupt_type, state=state
)
await sync_data(state=state, config=config, messages=[], trim=True)
return True
```

**Use cases**:
- Add approval metadata or SLA deadlines to `state.execution_meta.interrupt_data`
- Sanitize state before exposing/persisting an interrupted run
- Update frontend UI: change status from "thinking..." to "waiting for your approval"
- Send push notification to mobile app: "Agent needs your input"
- Store interrupt event in event log for auditability
- Start a timeout timer — if user doesn't resume within N minutes, auto-cancel
- Update a task queue entry with "PAUSED" status

**What can go wrong without it**:
- Frontend has to poll state to detect interrupts — wasteful and adds latency to UX
- No clean hook for "human-in-the-loop" workflow orchestration

---

### 5. `on_resume`

**Purpose**: Notify that a previously paused execution is continuing, and allow side effects before re-execution.

**Input**:
| Parameter | Type | Source |
|---|---|---|
| `context` | `GraphLifecycleContext` | Same context |
| `resumed_node` | `str` | `state.execution_meta.interrupted_node` — node being resumed |
| `state` | `AgentState` | The loaded interrupted state (before `state.clear_interrupt()`) |
| `resume_data` | `dict[str, Any]` | The `input_data` passed to `ainvoke()` on resume call |

**Output**: `AgentState | None`
- Return `None` → continue with the loaded interrupted state
- Return a modified `AgentState` → continue with that state after `clear_interrupt()`
- `resume_data` is intentionally mutable in place. The hook may normalize or enrich it before it is copied into `config["resume_data"]`.

**Fire location**: `InvokeHandler.invoke()` and `StreamHandler.stream()`, inside `check_interrupted()` when `state.is_interrupted()` is `True`, before `state.clear_interrupt()`.

> **Important**: This fires in `check_interrupted()` (the utility function), so the `CallbackManager` must be accessible there. Since `check_interrupted` is called with `@inject` dependencies, the `CallbackManager` can be injected via `Inject[CallbackManager]` the same way `BaseCheckpointer` is.

**Code point** (`heandler_utils.py`, inside `check_interrupted`):
```python
async def check_interrupted(state, input_data, config, callback_mgr=Inject[CallbackManager]):
    if state.is_interrupted():
        # NEW — fire before clear_interrupt():
        context = GraphLifecycleContext(
            config=config,
            timestamp=config.get("timestamp", ""),
        )
        state = await callback_mgr.fire_on_resume(
            context,
            resumed_node=state.execution_meta.interrupted_node or "",
            state=state,
            resume_data=input_data,
        )
        # ... existing logic
        config["resume_data"] = input_data
        state.clear_interrupt()
```

**Use cases**:
- Normalize/validate resume payload before execution continues
- Attach approval actor/timestamp to state before clearing the interrupt
- Log "execution resumed" with who approved and what data they sent
- Validate resume data before execution continues (e.g., check required fields)
- Update frontend status from "waiting" back to "thinking..."
- Cancel the timeout timer that was started in `on_interrupt`
- Audit trail: record the full interrupt → resume cycle with timestamps

**What can go wrong without it**:
- No visibility into the resume event — you can infer it from logs but not act on it cleanly

---

### 6. `on_checkpoint`

**Purpose**: React to state persistence events — replicate state, invalidate caches, trigger compliance logging.

**Input**:
| Parameter | Type | Source |
|---|---|---|
| `context` | `GraphLifecycleContext` | Same context |
| `state` | `AgentState` | The state object passed to `checkpointer.aput_state()` (may be context-trimmed) |
| `messages` | `list[Message]` | Messages passed to `checkpointer.aput_messages()` |
| `is_context_trimmed` | `bool` | Whether `context_manager.atrim_context()` was called before save |

**Output**: `tuple[AgentState, list[Message]] | AgentState | None`
- Return `None` → persist current `state` and `messages`
- Return `AgentState` → persist the returned state with current `messages`
- Return `(AgentState, list[Message])` → persist both returned values

**Fire location**: `utils/utils.py` — `sync_data()` function, after optional context trimming, before `checkpointer.aput_state()` and `checkpointer.aput_messages()`.

**Code point** (`utils.py`, inside `sync_data`):
```python
async def sync_data(state, config, messages, trim, checkpointer=Inject[...], ...):
    # ... existing logic ...
    new_state, messages = await callback_mgr.fire_on_checkpoint(
        context,
        state=new_state,
        messages=messages,
        is_context_trimmed=is_context_trimmed,
    )
    if checkpointer:
        await checkpointer.aput_state(config, new_state)
        if messages:
            await checkpointer.aput_messages(config, messages)
    return is_context_trimmed
```

**Note**: `sync_data()` is called in multiple places (during interrupt handling, on stop, and at graph end). `on_checkpoint` fires every time the runtime is about to persist state, not just at the end. This is intentional — checkpointing is an important event regardless of when it occurs.

**Use cases**:
- Redact or compact state immediately before durable persistence
- Drop transient messages that should not be checkpointed
- Replicate state to a secondary read-replica or CDN edge cache
- Invalidate frontend cache keyed by `thread_id` so next poll gets fresh state
- Write to a compliance audit log (required by SOC2, HIPAA)
- Mirror state to a different checkpointer backend (e.g., also write to Redis)
- Trigger a webhook to notify downstream systems of state change
- Record checkpoint frequency metrics

**What can go wrong without it**:
- No central point to enforce persistence-time policy without wrapping every checkpointer class

---

### 7. `on_state_update`

**Purpose**: Observe or modify state after each node transition — the most granular graph-level hook.

**Input**:
| Parameter | Type | Source |
|---|---|---|
| `context` | `GraphLifecycleContext` | Same context |
| `node_name` | `str` | `current_node` — the node that just executed |
| `old_state` | `AgentState` | A copy of state **before** the node ran (deep copy needed) |
| `new_state` | `AgentState` | State after the node result was merged |
| `step` | `int` | Current step number |

**Output**: `AgentState | None`
- Return `None` → use `new_state` unchanged
- Return a modified `AgentState` → replaces `new_state` for the rest of the graph

**Fire location**: `InvokeHandler._execute_graph()` and `StreamHandler._execute_graph()`, after result processing and state merging, before `call_realtime_sync()`.

> **Implementation note**: `old_state` must be captured as a `copy.deepcopy(state)` **before** `node.execute()` is called — otherwise both references point to the same mutated object. The deepcopy should be lightweight (using Pydantic's `.model_copy(deep=True)`).

**Code point** (`invoke_handler.py`, inside the while loop):
```python
# Before node execution — snapshot old state:
old_state_snapshot = state.model_copy(deep=True)

# ... node.execute(config, state) ...
# ... result processing / state merge ...

# NEW — after state update:
updated_state = await callback_mgr.fire_on_state_update(
    context,
    node_name=current_node,
    old_state=old_state_snapshot,
    new_state=state,
    step=step,
)
if updated_state is not None:
    state = updated_state

await call_realtime_sync(state, config)
```

**Use cases**:
- Real-time state diffing: compute what changed and stream diffs to frontend
- State validation: assert invariants after each node (e.g., required fields still present)
- Custom state merging: override how state fields are combined between nodes
- Live analytics: record per-node state metrics (context length growth, etc.)
- Security: scan state for sensitive data that should be masked before persisting
- Debug tooling: record full state history for replay debugging

**What can go wrong without it**:
- Debugging state corruption requires logs or breakpoints; no clean hook for state diffing
- Custom state validation must be baked into every node function instead of one centralized place

---

## Impact on Existing Code

| File | Change |
|---|---|
| `agentflow/utils/callbacks.py` | Add `GraphLifecycleContext`, `GraphLifecycleHook`, fire methods on `CallbackManager` |
| `agentflow/core/graph/utils/invoke_handler.py` | Inject `CallbackManager`, fire `on_graph_start`, `on_graph_end`, `on_graph_error`, `on_interrupt`, `on_state_update` |
| `agentflow/core/graph/utils/stream_handler.py` | Same as invoke_handler — both execution paths need hooks |
| `agentflow/core/graph/utils/heandler_utils.py` | Inject `CallbackManager`, fire `on_resume` in `check_interrupted()`, fire `on_interrupt` in `check_and_handle_interrupt()` and `interrupt_graph()` |
| `agentflow/core/graph/utils/utils.py` | Inject `CallbackManager`, fire `on_checkpoint` in `sync_data()` |
| `agentflow/utils/__init__.py` | Export `GraphLifecycleContext`, `GraphLifecycleHook` |

**Backward compatibility**: Zero breaking changes. All new hooks default to no-ops. `CallbackManager.__init__()` gets a new `_lifecycle_hooks: list[GraphLifecycleHook] = []` attribute. Existing `before_invoke` / `after_invoke` / `on_error` / validators are completely untouched.

---

## DI Integration Pattern

`CallbackManager` is already registered as a singleton in the DI container (`InjectQ`) and injected via `Inject[CallbackManager]` throughout the codebase (see `validate_message_content`, `InvokeNodeHandler`, `Node`). The same pattern applies to all new fire-point locations.

```python
from injectq import inject, Inject
from agentflow.utils.callbacks import CallbackManager

@inject  # or use Inject[CallbackManager] as default parameter
async def some_utility(
    state: StateT,
    config: dict,
    callback_mgr: CallbackManager = Inject[CallbackManager],
):
    ...
    await callback_mgr.fire_on_graph_start(context, state)
```

---

## Example Usage (End User API)

```python
from agentflow.utils.callbacks import GraphLifecycleHook, GraphLifecycleContext, CallbackManager
from agentflow.core.state import AgentState
from agentflow.core.state.message import Message

# ── Option A: Full lifecycle hook class ──────────────────────────────────────

class ObservabilityHook(GraphLifecycleHook):
    """Sends traces to OpenTelemetry and metrics to Prometheus."""

    async def on_graph_start(self, ctx: GraphLifecycleContext, state: AgentState):
        self._span = tracer.start_span(f"graph.run.{ctx.thread_id}")
        return None   # don't modify state

    async def on_graph_end(self, ctx, final_state, messages, total_steps):
        self._span.set_attribute("steps", total_steps)
        self._span.set_attribute("messages", len(messages))
        self._span.end()

    async def on_graph_error(self, ctx, error, partial_state, messages, step, node_name):
        self._span.record_exception(error)
        self._span.set_status(StatusCode.ERROR)
        self._span.end()
        alert_oncall(f"Graph failed at node {node_name}: {error}")
        return partial_state, str(error)

    async def on_checkpoint(self, ctx, state, messages, is_context_trimmed):
        metrics.increment("agentflow.checkpoints", tags={"thread": ctx.thread_id})

    async def on_interrupt(self, ctx, interrupted_node, interrupt_type, state):
        notify_frontend(ctx.thread_id, status="waiting_for_input", node=interrupted_node)

    async def on_resume(self, ctx, resumed_node, state, resume_data):
        notify_frontend(ctx.thread_id, status="resuming", node=resumed_node)

    async def on_state_update(self, ctx, node_name, old_state, new_state, step):
        diff = compute_diff(old_state, new_state)
        stream_diff_to_frontend(ctx.thread_id, diff)
        return None   # don't modify state


# ── Register with StateGraph ─────────────────────────────────────────────────

graph = StateGraph()
graph.add_node("research", research_agent)
graph.add_edge(START, "research")
graph.add_edge("research", END)

compiled = graph.compile()

# Register the hook via the callback manager
callback_mgr = CallbackManager()
callback_mgr.register_lifecycle_hook(ObservabilityHook())

# existing pattern still works:
callback_mgr.register_before_invoke(InvocationType.AI, my_ai_callback)
```

---

## Summary Table

| Hook | Returns | Can Modify? | Fires N times | Fire Location |
|---|---|---|---|---|
| `on_graph_start` | `AgentState \| None` | ✅ Yes (state) | 1 per run | After state load, before loop |
| `on_graph_end` | `AgentState \| None` | ✅ Yes (state) | 1 per successful run | After `state.complete()`, before final `sync_data()` |
| `on_graph_error` | `tuple[AgentState, str] \| None` | ✅ Yes (error snapshot) | 1 per failed run | In except block, before error `sync_data()` |
| `on_interrupt` | `AgentState \| None` | ✅ Yes (interrupt state) | 0-N per run | Before interrupt checkpoint persistence |
| `on_resume` | `AgentState \| None` | ✅ Yes (state + resume data in-place) | 0-1 per call | Before `clear_interrupt()` |
| `on_checkpoint` | `(AgentState, list[Message]) \| AgentState \| None` | ✅ Yes (persisted data) | 1-N per run | Before every durable checkpoint write |
| `on_state_update` | `AgentState \| None` | ✅ Yes (state) | N per run (once per node) | After each node result is merged |
