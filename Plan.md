# PyAgenity Tutorial Roadmap

## 🎯 Objectives
- Rebuild the content under `docs/Tutorial/` into a guided learning path rather than reference docs.
- Teach new users how to reason about PyAgenity’s building blocks (state, graph, nodes, tools, DI) before layering on persistence, streaming, and integrations.
- Tie every lesson to runnable code that ships in `examples/` so readers can clone, tweak, and verify behaviour.
- Keep parity with auto-generated API docs (from `scripts/generate_docs.py`) by leaving deep reference material there and focusing these tutorials on hands-on workflows and design decisions.

## 👩‍💻 Target Audience & Assumptions
- Python developers familiar with async/await and basic dependency injection patterns.
- New to PyAgenity but experienced with LLM APIs (LiteLLM, OpenAI, Gemini, etc.).
- Comfortable running `pytest`/scripts locally; expect to adjust `.env` with API keys.
- Wants prescriptive guidance on when to pick tools like ToolNode, custom states, or checkpointers.

## 📚 Tutorial Series Overview (single linear journey)
1. **Orientation & Setup** — explain install, environment variables, folder structure, InjectQ primer, quick glossary. *(`docs/Tutorial/index.md` as landing page)*
2. **Build Your First Agent Graph** — walk through `StateGraph` basics using `examples/react/react_sync.py`. Introduce START/END, nodes, edges, compile/invoke cycles. *(Rewrite `docs/Tutorial/graph.md` as a step-by-step lab.)*
3. **Crafting Message Flow & Agent State** — deep dive into `AgentState`, message schema, `ResponseGranularity`, helper utilities. Include customizing state subclass. *(Revamp `docs/Tutorial/state.md` and `docs/Tutorial/message.md` as a combined narrative.)*
4. **Injecting Tools & Dependency Resolution** — demonstrate ToolNode construction, DI with InjectQ, and how arguments (state, tool_call_id) auto-resolve. Leverage `examples/react/react_weather_agent.py` and `examples/react-injection/react_di.py`. *(Repurpose `docs/Tutorial/adapter.md` into “Tooling & Dependency Injection”.)*
5. **Branching, Routing, and Control Flow** — teach conditional edges, command-style returns, interrupt handling, and recursion limits. Reference `pyagenity/graph/utils/invoke_handler.py` concepts and the React agent pattern. *(Add a new section inside `graph.md` or a dedicated subsection.)*
6. **Persistence, Memory, and Stores** — introduce checkpointers (in-memory vs Postgres/Redis), stores, and when to attach them. Run through configuration snippets from `pyagenity/checkpointer/` & `pyagenity/store/`. *(Refocus `docs/Tutorial/checkpointer.md` and `store.md` on practical scenarios.)*
7. **Streaming, Events, and Observability** — explain `CompiledGraph.stream/astream`, `ResponseGranularity`, and publishers, highlighting console publisher and event payloads using `examples/react_stream/`. *(Turn `docs/Tutorial/publisher.md` & `callbacks.md` into a cohesive streaming/observability lab.)*
8. **Advanced Patterns & Prebuilt Agents** — showcase `pyagenity/prebuilt/agent/ReactAgent`, multi-agent orchestration (map/reduce, supervisor team), and optional extras (LiteLLM adapters, MCP, Composio, LangChain). Provide decision matrix for when to build vs use prebuilt. *(Use `docs/Tutorial/misc/` for advanced modules or create `docs/Tutorial/misc/advanced_patterns.md`.)*

## 🛠️ Content Tasks & File Mapping
- **Index Page** (`docs/Tutorial/index.md`)
	- New intro, prerequisites checklist, quickstart command block, and linkable roadmap.
	- Add callout boxes pointing to API reference and examples directory.
- **Graph Fundamentals** (`graph.md`)
	- Convert to a lab: scaffold project, add START→END node, compile, run, inspect output.
	- Explain Node execution contract, config dict, recursion guard.
- **State & Messages** (`state.md`, `message.md`)
	- Walk through `AgentState.context`, adding messages, customizing state with extra fields.
	- Include debugging tips (printing state, using ResponseGranularity.FULL).
- **Tools & DI** (`adapter.md` → rename section heading, optionally add `tooling.md` if needed)
	- Cover ToolNode local functions, MCP client handshake (`examples/react-mcp/`), dependency injection patterns (InjectQ container binding, autowired params).
- **Control Flow & Interrupts** (within `graph.md` or new `control_flow.md`)
	- Document `add_conditional_edges`, `Command`, interrupts, stop requests, recursion errors.
- **Persistence & Memory** (`checkpointer.md`, `store.md`)
	- Provide quick matrix of InMemory vs PgCheckpointer: when to use, configuration, required extras.
	- Include sample `.env` and code snippet wiring store + checkpointer into compile().
- **Streaming & Observability** (`publisher.md`, `callbacks.md`)
	- Step-by-step run of `stream_react_agent.py`, show sample streamed `EventModel` JSON.
	- Document callback registration patterns and best practices (validation, telemetry).
- **Advanced Patterns & Integrations** (`misc/advanced_patterns.md` new file)
	- Tour of prebuilt agents (React, Router, SupervisorTeam, MapReduce), when to adopt.
	- Outline optional dependencies from `pyproject.toml` and what features they unlock.

## ✍️ Writing Guidelines
- Keep each chapter actionable: context → code walkthrough → checkpoints → “try next”.
- Use admonitions for “When to use” vs “When to avoid”.
- Embed snippets referencing real files with `path/to/file.py` breadcrumbs.
- Include mini diagrams or ASCII flow summaries where helpful (no complex graphics required).
- Cross-link to auto-generated docs for API details instead of duplicating definitions.

## ✅ Acceptance & Validation
- Each tutorial references a runnable script in `examples/`; ensure code snippets align exactly with repository state.
- Provide “What to verify” checklists at the end of major lessons (e.g., expected console output, event log samples).
- Lint markdown with existing formatting style (80-100 char width, fenced code blocks labelled `python` or `bash`).
- After content updates, sanity-check the `scripts/generate_docs.py` flow still works (no structural surprises).

## 🚀 Implementation Sequence
1. Draft index & first-agent lesson to establish tone and confirm structure with maintainer.
2. Iterate through lessons 3–7, reusing example code and layering in DI, persistence, streaming.
4. Where possible, keep the refference to from `reference` and put into the tutorial files, so user want to learn in more details they can go to the reference docs.
3. Close with advanced patterns & optional extras, then add cross-links and navigation aids.
4. Run a final pass for terminology consistency (START/END constants, InjectQ naming) and update `Plan.md` with completion notes if needed.

---
*Status:* waiting for maintainer approval of this roadmap before editing tutorial files.
