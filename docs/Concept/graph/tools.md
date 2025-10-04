# Tools & Integrations

Tools extend an agent beyond pure language reasoning—letting it call functions, external APIs, local system utilities,
MCP servers, or third‑party registries like Composio or LangChain toolkits. PyAgenity unifies these via `ToolNode`.

---

## ToolNode Overview

`ToolNode` wraps one or more callables and presents them to the LLM as tool specifications. When the model emits
`tool_calls`, the graph routes to the ToolNode which executes each call and appends corresponding `tool` role messages.

Key responsibilities:

- Build JSON schemas for tool functions (signature inspection)
- Manage special injectable parameters (`tool_call_id`, `state`, etc.)
- Execute sequentially; aggregate results into messages
- Interleave with MCP / external tool sources

---

## Defining Local Tools

```python
from pyagenity.graph import ToolNode
from pyagenity.utils import Message
from pyagenity.state import AgentState

# Regular Python function
def get_weather(city: str, tool_call_id: str | None = None, state: AgentState | None = None) -> Message:
    data = {"city": city, "temp_c": 24}
    return Message.tool_message(content=data, tool_call_id=tool_call_id)

# Register
weather_tools = ToolNode([get_weather])
```

Return types allowed inside tools mirror node returns (Message, list[Message], str). Prefer `Message.tool_message()` for
clarity and structured result content.

---

## Injection in Tools

Special parameters auto-injected if present by name:

| Param | Meaning |
|-------|---------|
| `tool_call_id` | ID from originating assistant tool call block (pass through for traceability) |
| `state` | Current `AgentState` (read context or append additional messages) |

You can also inject container-bound dependencies using `Inject[Type]` just like nodes.

---

## Presenting Tools to the LLM

Tools are typically gathered right before an LLM completion call:

```python
if need_tools:
    tools = weather_tools.all_tools_sync()  # JSON schema list
    response = completion(model=model, messages=messages, tools=tools)
```

The LiteLLM converter then observes any resulting `tool_calls` in the response and the graph routes accordingly.

---

## MCP (Model Context Protocol) Tools

PyAgenity can integrate MCP tool providers (e.g. filesystem, Git, HTTP). MCP clients enumerate capabilities which the
ToolNode merges with local tools.

Conceptual steps:

1. Instantiate MCP client (outside scope here)
2. Pass MCP-derived tool specs into ToolNode or merge at invocation time
3. During tool execution dispatch through MCP client

A future high-level helper will streamline this; current pattern mirrors local tool injection with an adapter.

---

## Composio Integration

Composio offers a catalogue of real-world service connectors. An adapter (see `pyagenity/adapters/tools/composio_adapter.py` if present) maps Composio tool manifests into the ToolNode schema format.

Benefits:

- Avoid hand-writing repetitive API wrappers
- Centralised auth management
- Standardised parameter JSON schema

---

## LangChain Tools

For teams already using LangChain, you can register LangChain tool objects via the LangChain adapter. It converts
LangChain tool metadata into a shape PyAgenity expects (name, description, parameters). Mixed usage with local tools is
supported.

---

## Tool Execution Flow

```
Assistant (LLM) → tool_calls[] → Graph edge → ToolNode
ToolNode:
  for each tool_call:
     locate matching tool
     prepare args
     inject (tool_call_id, state, deps)
     execute
     collect result → Message.role=tool
Return tool messages → appended to state.context → next node
```

---

## Error Handling in Tools

Recommendations:

- Raise exceptions for unrecoverable errors; upstream can decide to retry
- Return structured error payloads (e.g. `{ "error": "timeout" }`) for model-readable handling
- Log via injected `CallbackManager` / `Publisher` for observability

---

## Caching & Idempotency

If tool calls are expensive (e.g. web search), consider:

- Injecting a cache service (Redis / in-memory) to memoize `(tool_name, args)`
- Adding a hash of arguments into tool result metadata
- Storing results in `state` for reuse in later reasoning steps

---

## Security Considerations

| Area | Risk | Mitigation |
|------|------|------------|
| Shell / OS tools | Arbitrary command execution | Maintain allow-list, sandbox execution |
| External APIs | Credential leakage | Store keys in DI container with least privilege |
| MCP file access | Sensitive file exfiltration | Restrict path roots, enforce read-only mode |
| Tool arguments | Prompt injection into tool layer | Validate & sanitize inputs |

---

## Testing Tools

```python
def test_get_weather():
    msg = get_weather("Paris", tool_call_id="abc")
    assert msg.role == "tool"
    assert msg.metadata.get("tool_call_id") == "abc" if msg.metadata else True
```

For batch tool calls simulate the `tools_calls` structure and invoke the ToolNode directly.

---

## Best Practices

- Keep tool interfaces narrow: specific verbs beat generic catch-alls
- Use structured outputs (dicts) instead of raw strings whenever feasible
- Provide short, action-focused descriptions (improves model selection accuracy)
- Constrain argument types—avoid `any` shaped blobs unless necessary

---

Next: Execution Runtime (`execution.md`).
