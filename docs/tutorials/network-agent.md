---
title: Network Agent
---

# Network Agent

The Network pattern lets you define an arbitrary set of nodes and route between them with static or conditional edges.

Example:

```python
from pyagenity.prebuilt.agent import NetworkAgent
from pyagenity.utils import Message

def n1(state, config):
    return Message.create(role="assistant", content="n1")

def n2(state, config):
    return Message.create(role="assistant", content="n2")

agent = NetworkAgent()
app = agent.compile(
    nodes={"A": n1, "B": n2},
    entry="A",
    static_edges=[("A", "B")],
)

out = app.invoke({"messages": [Message.from_text("go")]}, config={"thread_id": "net"})
```
