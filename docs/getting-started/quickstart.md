---
title: Quickstart
---

# Quickstart

A minimal example showing a simple graph with one node.

```python
from pyagenity.graph import StateGraph, Node


class MyNode(Node):
    def run(self, state: dict) -> dict:
        name = state.get("name", "world")
        return {"message": f"Hello, {name}!"}


graph = StateGraph()
graph.add_node("greeter", MyNode())
graph.set_entry_point("greeter")

result = graph.invoke({"name": "PyAgenity"})
print(result["message"])  # -> Hello, PyAgenity!
```

Next, explore the Guides for deeper topics like callbacks, streaming, and pause/resume.
