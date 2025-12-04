# Realtime Support Plan for Agentflow

This document outlines the plan to integrate real-time, bidirectional streaming capabilities into `agentflow`, leveraging the Google Gemini Live API.

## Objective
Enable `agentflow` graphs to support real-time voice and text interaction, where the agent can receive a continuous stream of inputs (audio/text) and produce a continuous stream of outputs (audio/text), while maintaining the ability to execute tools defined in the graph.

## Architecture

The implementation will introduce a new `realtime` module to `agentflow` that interfaces directly with the `google-genai` SDK (and potentially others in the future) to handle the WebSocket-based Live API.

### 1. New Module: `agentflow.realtime`

This module will contain the core logic for managing live sessions.

#### `LiveRequestQueue`
A queue-based class to buffer inputs from the client (e.g., a WebSocket server) to be sent to the model.
- **Methods**:
    - `put_text(text: str)`
    - `put_audio(audio_data: bytes)`
    - `get()`: Async generator yielding inputs.

#### `RealtimeSession`
A class that manages the lifecycle of a Gemini Live session.
- **Responsibilities**:
    - Connect to the Gemini Live API using `google.genai.Client`.
    - Convert `agentflow` tool definitions (from `ToolNode`) to Google GenAI compatible schemas.
    - Manage the bidirectional loop:
        - **Send Loop**: Consumes `LiveRequestQueue` and sends to the model.
        - **Receive Loop**: Consumes model events.
            - Yields audio/text content to the caller.
            - Intercepts `tool_call` events.
            - Executes tools using the graph's `ToolNode`.
            - Sends `tool_response` back to the model.

### 2. Integration with `CompiledGraph`

We will extend `CompiledGraph` to support a new execution mode: `run_live`.

```python
class CompiledGraph:
    # ... existing code ...

    async def run_live(
        self,
        input_queue: LiveRequestQueue,
        config: dict[str, Any] | None = None,
        tool_node_name: str = "tools"
    ) -> AsyncIterator[LiveEvent]:
        """
        Starts a realtime session with the compiled graph.
        """
        # Implementation details...
```

### 3. Tool Schema Conversion
Since `agentflow` generates OpenAI-compatible schemas, we need a utility to convert them to the format expected by `google-genai` (if different, or ensure compatibility).

## Implementation Steps

### Step 1: Add Dependencies
- Add `google-genai` as an optional dependency in `pyproject.toml`.

### Step 2: Create `agentflow/realtime` Structure
- Create `agentflow/realtime/__init__.py`
- Create `agentflow/realtime/queue.py` (`LiveRequestQueue`)
- Create `agentflow/realtime/session.py` (`RealtimeSession`)
- Create `agentflow/realtime/utils.py` (Schema conversion)

### Step 3: Implement `LiveRequestQueue`
- Simple async queue wrapper.

### Step 4: Implement `RealtimeSession`
- **Init**: Takes `model`, `tools` (from `ToolNode`), `system_instruction`.
- **Connect**: Uses `client.aio.live.connect`.
- **Loop**: Implements the send/receive logic with tool execution.
- **Tool Execution**: Needs to resolve the tool function from `ToolNode` and execute it.

### Step 5: Update `CompiledGraph`
- Add `run_live` method in `agentflow/graph/compiled_graph.py`.
- This method will:
    1.  Identify the `ToolNode` (by name or type).
    2.  Extract tools.
    3.  Initialize `RealtimeSession`.
    4.  Return the session's event stream.

## Usage Example

```python
# User side (e.g., FastAPI)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create input queue
    input_queue = LiveRequestQueue()
    
    # Start agent session
    # graph is a CompiledGraph
    live_events = graph.run_live(input_queue, config={"model": "gemini-2.0-flash-exp"})
    
    # Tasks to handle I/O
    async def receive_from_client():
        while True:
            data = await websocket.receive()
            # push to input_queue
            
    async def send_to_client():
        async for event in live_events:
            # send to websocket
            
    await asyncio.gather(receive_from_client(), send_to_client())
```

## Future Considerations
- Support for other providers (OpenAI Realtime) via LiteLLM if/when fully supported.
- Session persistence/resumption (ADK feature).
