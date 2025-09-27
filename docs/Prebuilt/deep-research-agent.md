# Deep Research Agent

The DeepResearchAgent provides a structured loop for long-horizon, information-seeking tasks. It follows a PLAN → RESEARCH → SYNTHESIZE → CRITIQUE cycle with bounded iterations and an optional heavy mode, inspired by contemporary deep research agents.

## When to use

- Multi-hop web/code research
- Literature review and fact-checking
- Generating comprehensive reports from multiple sources

## Topology

- PLAN: decomposes the task and may request tools
- RESEARCH: executes tool calls (search, crawl, calc, etc.)
- SYNTHESIZE: aggregates findings into a coherent draft
- CRITIQUE: detects gaps/contradictions and decides whether to iterate or finish

Edges:
- PLAN → RESEARCH or SYNTHESIZE (conditional)
- RESEARCH → SYNTHESIZE
- SYNTHESIZE → CRITIQUE
- CRITIQUE → RESEARCH or END (conditional)

Iteration control uses `execution_meta.internal_data` keys:
- `dr_max_iters` (int, default 2)
- `dr_iters` (int, auto-updated)
- `dr_heavy_mode` (bool, default False)

## Usage

```python
from pyagenity.prebuilt.agent import DeepResearchAgent
from pyagenity.graph.tool_node import ToolNode
from pyagenity.utils.message import Message

def plan(state, config):
    # produce an assistant message with optional tool calls
    return Message.create(role="assistant", content="Planning done.")

def synthesize(state, config):
    return Message.create(role="assistant", content="Synthesis done.")

def critique(state, config):
    # can optionally include tool calls to trigger more research
    return Message.create(role="assistant", content="Critique done.")

def search_tool(q: str) -> str:
    return f"search:{q}"

tools = ToolNode([search_tool])

agent = DeepResearchAgent(max_iters=2, heavy_mode=False)
app = agent.compile(plan_node=plan, research_tool_node=tools, synthesize_node=synthesize, critique_node=critique)

res = app.invoke({"messages": [Message.from_text("Find X")]})
```

## Notes

- Provide deterministic tools in tests for reproducibility.
- You can seed `dr_max_iters` and `dr_heavy_mode` when constructing the agent.
