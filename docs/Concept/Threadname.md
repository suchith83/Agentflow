## Thread Name Generation

Thread names label an execution thread (session) for correlation across logs, publishers, and external APIs. While IDs are great for machines, a human-friendly thread name helps during debugging, dashboards, and interactive UIs.

Implemented in: `pyagenity/utils/thread_name_generator.py`

### How It Works

`StateGraph` accepts an optional `thread_name_generator` callable. If not provided, the framework uses `generate_dummy_thread_name()` which delegates to `AIThreadNameGenerator` producing names like:

```
thoughtful-dialogue
creative-exploration
focused-analysis
```

These come from adjective–noun / action / compound pattern pools.

### API Surface

```python
from pyagenity.utils.thread_name_generator import (
	generate_dummy_thread_name,
	AIThreadNameGenerator,
)

name = generate_dummy_thread_name()  # one-off helper
gen = AIThreadNameGenerator()
custom = gen.generate_action_name()
```

`AIThreadNameGenerator` exposes:

- `generate_simple_name()` – adjective-noun pairs
- `generate_action_name()` – action + object (e.g. `exploring-ideas`)
- `generate_compound_name()` – compound descriptors (e.g. `deep-dive`)
- `generate_name()` – strategy rotation across patterns

### Attaching to a Graph

```python
from pyagenity.graph import StateGraph
from pyagenity.utils.thread_name_generator import AIThreadNameGenerator


def handler(state, config):
	return []

graph = StateGraph(thread_name_generator=AIThreadNameGenerator().generate_name)
graph.add_node("MAIN", handler)
graph.add_edge("MAIN", "__end__")
graph.set_entry_point("MAIN")
app = graph.compile()
```

When deployed via a future PyAgenity API service layer (PyAgnoty-API), a thread name is auto-generated if none is supplied, aiding multi-session inspection.

### Overriding Strategy

Custom generator example:

```python
import random

TOPICS = ["pricing", "support", "onboarding", "research"]

def ticket_style_thread_name():
	import uuid
	return f"tkt-{random.choice(TOPICS)}-{str(uuid.uuid4())[:8]}"

graph = StateGraph(thread_name_generator=ticket_style_thread_name)
```

### Accessing the Name

At runtime the thread name is placed into execution metadata / config (e.g. `config["thread_id"]` you pass, or generated internally). Combine a human thread name with a machine ID for best traceability.

### When to Use Custom Naming

| Use Case | Benefit |
|----------|---------|
| Customer support sessions | Easier triage (e.g. `support-session`) |
| Multi-agent dashboards | Human-scannable grouping |
| A/B test cohorts | Encode variant in name (`variantA-dialogue`) |
| Sharded processing | Prefix shard to aid log routing |

### Pitfalls

- Avoid personally identifiable information in names.
- Very long names can clutter dashboards; keep under ~40 chars.
- Ensure uniqueness if names are used as keys; otherwise combine with `generated_id`.

### Testing

Stub the generator for deterministic snapshots:

```python
graph = StateGraph(thread_name_generator=lambda: "test-thread")
```

---

See also: `ID Generation` (numeric/string IDs) and `Response Converter` (message normalisation).
