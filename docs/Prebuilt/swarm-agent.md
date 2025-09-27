---
title: Swarm Agent
---

# Swarm Agent

The Swarm pattern broadcasts work across multiple worker nodes, optionally collects intermediate results, and then produces a final consensus.

Key nodes:
- DISPATCH (optional): prepare or plan work
- WORKER_i: one or more worker functions or tool nodes
- COLLECT (optional): consolidate intermediate results (applied per worker)
- CONSENSUS: aggregate and finalize the answer

Example:

```python
from pyagenity.prebuilt.agent import SwarmAgent
from pyagenity.utils import Message

def worker_a(state, config):
    return Message.create(role="assistant", content="worker_a")

def worker_b(state, config):
    return Message.create(role="assistant", content="worker_b")

def collect(state, config):
    return Message.create(role="assistant", content="collect")

def consensus(state, config):
    return Message.create(role="assistant", content="consensus")

agent = SwarmAgent()
app = agent.compile(
    workers={"A": worker_a, "B": worker_b},
    consensus_node=consensus,
    options={"collect": collect},
)

out = app.invoke({"messages": [Message.from_text("run")]}, config={"thread_id": "swarm"})
```
