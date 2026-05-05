# StreamEmitter Plan

## Goal

Add a `StreamEmitter` that is automatically injected into tool calls during streaming execution. Tools can use it to emit progress, status, message, and error updates to the same `app.stream(...)` / `app.astream(...)` output that the frontend already consumes.

This feature must have no effect on normal `invoke(...)` / `ainvoke(...)` calls.

## Problem

Today, tool execution can publish events through the publisher system, but frontend clients usually cannot consume Redis, Kafka, RabbitMQ, or other backend publishers directly.

Also, a local tool such as `get_weather(...)` runs like a normal function. If it retries internally, the graph stream does not see those intermediate retry states. The frontend only receives the final tool result.

The desired behavior is:

1. A tool starts running in streaming mode.
2. The tool emits progress while it runs.
3. The frontend receives those updates through the existing graph stream.
4. The final tool result is still returned normally.
5. Non-streaming invoke behavior remains unchanged.

## API Shape

Tools should be able to accept an optional injected `emit` parameter:

```python
def get_weather(
    location: str,
    tool_call_id: str | None = None,
    state: CustomAgentState | None = None,
    emit: StreamEmitter | None = None,
) -> str:
    ...
```

Example usage:

```python
if emit:
    emit.progress(
        "Attempt 1 failed. Retrying...",
        data={"attempt": 1, "max_attempts": 3},
    )
```

Final failure can be emitted as:

```python
if emit:
    emit.error(
        "Weather lookup failed after 3 attempts.",
        data={"location": location},
    )
```

## StreamEmitter Responsibilities

Create a class called `StreamEmitter`.

It should provide sync-friendly methods:

```python
emit.progress(message: str, data: dict | None = None) -> None
emit.error(message: str, data: dict | None = None) -> None
emit.message(message: str, data: dict | None = None) -> None
emit.update(data: dict) -> None
```

The methods should create `StreamChunk` objects and push them into the active streaming output path.

The emitted chunks should include useful metadata:

```python
{
    "status": "tool_progress",
    "message": "...",
    "tool_name": "get_weather",
    "tool_call_id": "call_abc",
    "node": "TOOL",
    "attempt": 1,
    "max_attempts": 3,
}
```

## Streaming Behavior

During `app.stream(...)` / `app.astream(...)`:

1. `StreamNodeHandler` detects a tool call.
2. It creates a queue for emitted stream chunks.
3. It creates a `StreamEmitter` bound to that queue and tool context.
4. It passes the emitter into `ToolNode.stream(...)`.
5. `ToolNode.stream(...)` passes the emitter into `_internal_execute(...)`.
6. `_internal_execute(...)` injects `emit` into the tool function if requested.
7. While the tool runs, emitted chunks are yielded to the graph stream.
8. When the tool finishes, the normal tool result is yielded as before.

## Invoke Behavior

Regular `invoke(...)` / `ainvoke(...)` should not create or inject an active emitter.

Options:

1. Do not inject `emit` in the invoke path.
2. Inject `emit=None` in the invoke path.
3. Inject a no-op emitter in the invoke path.

Recommended: inject `emit=None` or do not inject it at all for invoke. This makes it clear that custom streaming updates are only available in streaming mode.

## Files To Update

### `agentflow/core/state/stream_emitter.py`

Add the new `StreamEmitter` class.

It should know:

```python
tool_name
tool_call_id
node_name
thread_id
run_id
queue        # asyncio.Queue
loop         # asyncio.AbstractEventLoop, captured at construction
```

It should build `StreamChunk` values using:

```python
StreamEvent.UPDATES
StreamEvent.ERROR
StreamEvent.MESSAGE
```

**Thread safety:** All emit methods must use `loop.call_soon_threadsafe(queue.put_nowait, chunk)` instead of `queue.put_nowait(chunk)` directly. Sync tools run inside `asyncio.to_thread(...)` on a thread-pool thread; calling `queue.put_nowait` directly from that thread is not safe. `call_soon_threadsafe` schedules the put on the event loop from any thread.

### `agentflow/core/state/__init__.py`

Export `StreamEmitter`.

### `agentflow/core/graph/tool_node/constants.py`

Add `"emit"` to injectable parameters so it is excluded from generated LLM tool schemas.

### `agentflow/core/graph/tool_node/base.py`

Extend `ToolNode.stream(...)` to accept:

```python
emit: StreamEmitter | None = None
```

Place `emit` **before** the DI-injected `callback_manager` parameter, since parameters with defaults must not follow DI-sentinel parameters in a way that breaks call sites:

```python
async def stream(
    self,
    name: str,
    args: dict,
    tool_call_id: str,
    config: dict,
    state: AgentState,
    emit: StreamEmitter | None = None,          # ← before DI param
    callback_manager: CallbackManager = Inject[CallbackManager],
) -> AsyncIterator[...]:
```

Pass `emit` into `_internal_execute(...)` only for the `_funcs` branch. Do not change `ToolNode.invoke(...)` behavior.

### `agentflow/core/graph/tool_node/executors.py`

Update `_internal_execute(...)` to accept:

```python
emit: StreamEmitter | None = None
```

Add it to the injectable defaults dict passed to `_prepare_input_data_tool`:

```python
{
    "tool_call_id": tool_call_id,
    "state": state,
    "config": config,
    "emit": emit,
}
```

Update `_prepare_input_data_tool(...)` so `"emit"` is handled alongside `"state"`, `"config"`, and `"tool_call_id"` in the explicit injection check:

```python
if param_name in ["state", "config", "tool_call_id", "emit"]:
    input_data[param_name] = default_data[param_name]
    continue
```

**Important:** `"emit"` must appear in this explicit check, not only in `INJECTABLE_PARAMS`. The `INJECTABLE_PARAMS` set causes params to be skipped entirely (using their default value). The explicit check injects the actual value from `default_data`. Adding `"emit"` only to `INJECTABLE_PARAMS` would mean the tool always receives `None` even in streaming mode.

**Note:** `emit` injection applies only to local (`_funcs`) tools. MCP, Composio, and LangChain tool paths are out of scope and are unchanged.

### `agentflow/core/graph/utils/stream_node_handler.py`

Create the emitter in `_handle_single_tool(...)` and pass it to `ToolNode.stream(...)`.

The current parallel tool implementation uses `asyncio.gather(...)`, which collects results before yielding them. To support live emitted chunks, replace that buffering behavior with an output queue pattern so chunks can be yielded as soon as tools emit them.

## Queue Pattern

For each tool call:

1. Create a per-tool `asyncio.Queue()`.
2. Create a `StreamEmitter` bound to that queue.
3. Run `ToolNode.stream(...)` in a background `asyncio.Task`.
4. The emitter pushes `StreamChunk`s into the queue via `loop.call_soon_threadsafe`.
5. The background task also pushes final tool results into the queue.
6. When the background task completes, it pushes a sentinel value (`_DONE`) into the queue.
7. `_handle_single_tool(...)` yields queue items until it sees the sentinel, then stops.

For multiple parallel tool calls:

1. Each tool has its own emitter and its own per-tool queue.
2. `_call_tools(...)` uses a single shared output queue and `N` background workers.
3. Each worker drains its tool's generator and pushes results to the shared queue.
4. Each worker pushes a sentinel when done.
5. `_call_tools(...)` yields from the shared queue until all `N` sentinels are received.

This replaces the current `asyncio.gather` + full-buffer pattern, allowing chunks from one tool to be yielded immediately without waiting for all other tools to finish.

**Exception safety:** Each background task must push the sentinel in a `finally` block so that an exception inside a tool does not leave the consumer loop hanging forever. The exception should be re-raised after the consumer has finished draining.

## Publisher Integration

Publisher integration is **out of scope for the initial implementation**.

The primary contract for `StreamEmitter` is the graph stream (the `asyncio.Queue` path). Frontend delivery must work without Redis, Kafka, RabbitMQ, or any other backend publisher.

In a future iteration, `StreamEmitter` may optionally mirror emitted chunks to `publish_event(...)` for backend monitoring. The interface should be designed so this can be added without breaking existing users.

## Example Frontend Stream Events

Progress:

```json
{
  "event": "updates",
  "data": {
    "status": "tool_progress",
    "message": "Attempt 1 failed. Retrying...",
    "tool_name": "get_weather",
    "tool_call_id": "call_abc",
    "attempt": 1,
    "max_attempts": 3
  }
}
```

Error:

```json
{
  "event": "error",
  "data": {
    "status": "tool_failed",
    "message": "Weather lookup failed after 3 attempts.",
    "tool_name": "get_weather",
    "tool_call_id": "call_abc"
  }
}
```

## Acceptance Criteria

1. A streaming tool can receive an injected `emit` parameter.
2. Calling `emit.progress(...)` inside the tool produces a `StreamChunk` in `app.stream(...)`.
3. Calling `emit.error(...)` inside the tool produces an error chunk in `app.stream(...)`.
4. Final tool results still behave as they do today.
5. `invoke(...)` and `ainvoke(...)` behavior remains unchanged.
6. `emit` does not appear in generated tool schemas.
7. Parallel tool calls can emit updates without waiting for all tools to finish.
8. Existing publisher behavior continues to work.

## Testing Plan

Add tests for:

1. Tool schema generation excludes `emit`.
2. `app.stream(...)` receives emitted progress chunks from a tool.
3. `app.stream(...)` receives emitted error chunks from a tool.
4. `app.invoke(...)` still works without requiring `emit`.
5. Multiple parallel tools can emit chunks and final results.
6. Existing tool result messages still merge into state correctly.

