# PyAgenity

PyAgenity is a lightweight Python framework for building intelligent agents and orchestrating multi-agent workflows on top of the LiteLLM unified LLM interface.

Core features:

- Unified `Agent` abstraction (no raw LiteLLM objects leaked)
- Structured responses with `content`, optional `thinking`, and `usage`
- Streaming support with incremental chunks
- Final message hooks for persistence/logging
- LangGraph-inspired graph engine: nodes, conditional edges, pause/resume (human-in-loop)
- In-memory session state store (pluggable in the future)

## Installation

```bash
pip install -e .
```

Set provider API keys (example for OpenAI):

```bash
export OPENAI_API_KEY=sk-...  # required for gpt-* models
```

`.env` is auto-loaded if present (via `python-dotenv`).

## Basic Agent Usage

```python
from pyagenity.agent.agent import Agent

agent = Agent(name="demo", model="gpt-4o-mini")
resp = agent.run("Hello there!")
print(resp.content)
print(resp.usage)
```

### Streaming

```python
for chunk in agent.run("Stream this response", stream=True):
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
```

## Graph Orchestration

```python
from pyagenity.graph import Graph, Edge, LLMNode, HumanInputNode, FunctionNode, GraphExecutor, SessionStatus
from pyagenity.agent.agent import Agent

summarizer = Agent(name="summarizer", model="gpt-4o-mini")
critic = Agent(name="critic", model="gpt-4o-mini")

def summarize_builder(state):
    return f"Summarize: {state.get('text','')}"

def critique_builder(state):
    return f"Critique this summary: {state.get('last_response','')}"

def combine(state):
    state['final'] = f"Improved: {state.get('critique_response','')}"
    return state

g = Graph()
g.add_node(LLMNode(name="summarize", agent=summarizer, prompt_builder=summarize_builder), start=True)
g.add_node(LLMNode(name="critique", agent=critic, prompt_builder=critique_builder, output_key="critique_response"))
g.add_node(HumanInputNode(name="human_review"))
g.add_node(FunctionNode(name="finalize", func=combine))

g.add_edge(Edge("summarize", "critique"))
g.add_edge(Edge("critique", "human_review"))
g.add_edge(Edge("human_review", "finalize", condition=lambda s: s.get("human_input") is not None))

executor = GraphExecutor(g)
state = executor.start({"text": "PyAgenity is a framework for multi-agent orchestration."})
if state.status == SessionStatus.WAITING_HUMAN:
    state = executor.resume(state.session_id, human_input="Looks good; proceed.")
print(state.status, state.shared.get("final"))
```

See `example/graph_demo.py` for a runnable example.

## Human-in-the-Loop

A `HumanInputNode` causes execution to pause (`WAITING_HUMAN`) if `human_input` is absent. Provide input via `resume(session_id, human_input=...)`.

## Final Hooks

Use `GraphExecutor.add_final_hook(callable)` to register hooks invoked when a session reaches `COMPLETED` or `FAILED`.

## Roadmap

- Persistent state backend (Redis, SQL, etc.)
- Parallel / branching strategies and selection policies
- Tool invocation nodes & function calling wrappers
- Tracing / telemetry integration

## License

MIT
