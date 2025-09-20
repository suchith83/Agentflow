# Copilot instructions for PyAgenity

Use these repo-specific notes to be productive quickly when generating code, docs, or tests.

## Big picture
- PyAgenity is a lightweight Python framework for building agent graphs on top of LiteLLM.
- Core primitives live in `pyagenity/graph/`: `StateGraph`, `Node`, `Edge`, `ToolNode`, `CompiledGraph`.
- State is a Pydantic model (`state/agent_state.py`); messages are `utils/message.py::Message`.
- Flow: build a `StateGraph` → add nodes/edges (incl. conditional) → `set_entry_point` → `compile()` → use `CompiledGraph.invoke()` or `CompiledGraph.stream()`.
- START/END constants come from `utils/constants.py` (`START="__start__"`, `END="__end__"`). Always reference these.

## Architecture essentials
- Graph API (see `graph/state_graph.py`, `graph/compiled_graph.py`, `graph/node.py`):
  - `add_node(name, func|ToolNode)`, `add_edge(from, to)`, `add_conditional_edges(from, condition, path_map)`.
  - `compile()` wires DI (InjectQ), checkpointer, callbacks, publisher; default checkpointer is in-memory.
- Node contracts (`graph/node.py`):
  - Functions usually accept `(state: AgentState, config: dict, …deps)` and return updated state or `list[Message]`.
  - Streaming supported by `.stream(...)` and `EventModel` chunks.
- Tools via `ToolNode` (`graph/tool_node/`):
  - Register plain callables or integrate MCP, Composio, LangChain. Specs via `await tool_node.all_tools()`.
  - Tool functions may declare injectable params like `tool_call_id` and `state` (see README example).
- Persistence & events:
  - Checkpointers: `InMemoryCheckpointer` (default) and `PgCheckpointer` (Postgres+Redis, extras required).
  - Publishers emit execution events (`ConsolePublisher` for dev). Events are `utils/streaming.py::EventModel`.

## Authoritative examples
- Minimal graph pattern: `tests/graph/test_graph.py` shows `add_node` → `set_entry_point` → `add_edge(..., END)` → `compile()` → `invoke()`.
- Tooling + routing + LiteLLM: `examples/react/react_weather_agent.py`.
- MCP tools: `examples/react-mcp/`.
- Prebuilt agents: `pyagenity/prebuilt/agent/` (React, RAG, Swarm, Router, MapReduce, SupervisorTeam, etc.).

## Conventions
- Create messages via `Message.from_text(...)`, `Message.tool_message(...)`, or `Message.from_response(...)`.
- ID generation is DI-driven: keys `generated_id_type` ("string"|"int"|"bigint") and optional `generated_id` (value/awaitable). See `utils/message.py` and `utils/id_generator.py`.
- Conditional routing returns labels that must match `path_map` keys; include `END` to terminate.
- Tests are pytest-only from `tests/` (config in `pyproject.toml`; markers: `asyncio`, `integration`; coverage enabled).

## Workflows
- Setup: Python ≥ 3.10. Dev deps in `requirements-dev.txt`. `.env` auto-loaded via `python-dotenv`.
- Tests: `make test` (uv) or VS Code task “pytest -q”. Coverage: `make test-cov`.
- Build/publish: `make build`, `make publish` (or `make testpublish`).
- Docs: `make docs-serve` (mkdocs at http://127.0.0.1:8000), `make docs-build`.
- Examples need an LLM key (e.g., `OPENAI_API_KEY`) and a valid LiteLLM model string.

## Optional integrations (extras in `pyproject.toml`)
- `pg_checkpoint`: Postgres (`asyncpg`) + Redis cache (`redis`); use `checkpointer.PgCheckpointer`.
- `mcp`: Model Context Protocol (FastMCP + mcp); pass a client to `ToolNode`.
- `composio`, `langchain`: adapters for external tool registries via `ToolNode`.
- Publishers: `redis`, `kafka`, `rabbitmq` classes in `pyagenity/publisher/`.

## Gotchas
- Call `set_entry_point(...)` before `compile()` or you’ll get `GraphError`.
- In `add_conditional_edges`, ensure the condition’s return values match `path_map` keys exactly (including `END`).
- For `PgCheckpointer`, install extras and run `setup()/asetup()` to initialize schema before use.
- `CompiledGraph.stream/astream` yields `EventModel` chunks; use `content_type` to parse.

## Environment
activate using `source .venv/bin/activate`
and run pytest from the project root.
```
pytest
```