# Custom State Usage Guide for PyAgenity

## Overview
PyAgenity supports custom state classes that extend the base `AgentState` class. This allows you to add application-specific fields while maintaining compatibility with the framework's execution, persistence, and checkpointing features.

## Creating Custom States

### Method 1: Using @dataclass decorator (Recommended)
When you need complex field initialization (like lists, dicts with defaults):

```python
from dataclasses import dataclass, field
from typing import Any, List
from pyagenity.state.agent_state import AgentState

@dataclass
class MyCustomState(AgentState):
    """Custom state with complex field types."""
    items: List[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    count: int = 0
    name: str = "default"
```

### Method 2: Simple class inheritance
When you only need simple default values:

```python
from pyagenity.state.agent_state import AgentState

class MySimpleState(AgentState):
    """Custom state with simple fields."""
    candidate_cv: str = ""
    jd: str = ""  # job description
    match_score: float = 0.0
```

## Graph Construction

### With Custom State
```python
from pyagenity.graph import StateGraph
from pyagenity.checkpointer import InMemoryCheckpointer

# Create graph with custom state type
graph = StateGraph[MyCustomState](MyCustomState())

# Create checkpointer with custom state type
checkpointer = InMemoryCheckpointer[MyCustomState]()
app = graph.compile(checkpointer=checkpointer)
```

### With Default AgentState
```python
# When no custom state is needed
graph = StateGraph()  # Uses AgentState by default
```

## Node Functions

Node functions should accept your custom state type:

```python
async def my_node(state: MyCustomState, config: dict[str, Any]) -> MyCustomState:
    """Node that works with custom state."""
    # Modify state fields
    state.count += 1
    state.items.append(f"item_{state.count}")
    state.metadata["processed"] = True
    
    # Return the modified state
    return state
```

## Key Rules

1. **@dataclass Required for field()**: If you use `field(default_factory=...)`, you MUST use the `@dataclass` decorator
2. **AgentState is Already a Dataclass**: The base `AgentState` class is already decorated with `@dataclass`
3. **Type Hints Required**: Always use proper type hints for your custom fields
4. **Graph Type Parameter**: Use `StateGraph[YourStateType](instance)` for type safety
5. **Checkpointer Type**: Use `InMemoryCheckpointer[YourStateType]()` for proper state persistence

## Common Patterns

### CV/Resume Analysis State
```python
@dataclass
class CVAnalysisState(AgentState):
    candidate_cv: str = ""
    job_description: str = ""
    match_score: float = 0.0
    analysis_results: dict = field(default_factory=dict)
    skills_found: List[str] = field(default_factory=list)
```

### Chat Bot State
```python
@dataclass
class ChatBotState(AgentState):
    user_preferences: dict = field(default_factory=dict)
    conversation_stage: str = "greeting"
    collected_info: dict = field(default_factory=dict)
```

### Multi-Step Process State
```python
@dataclass
class ProcessState(AgentState):
    steps_completed: List[str] = field(default_factory=list)
    current_step: str = "init"
    step_results: dict = field(default_factory=dict)
    error_count: int = 0
```

## Troubleshooting

### Field Objects Instead of Values
**Problem**: Getting `dataclasses.Field` objects instead of actual values
**Solution**: Ensure you use `@dataclass` decorator when using `field(default_factory=...)`

### TypeError with isinstance
**Problem**: `isinstance() arg 2 must be a type or tuple of types`
**Solution**: Fixed in framework - use proper string annotations for TypeVar bounds

### State Not Persisting
**Problem**: Custom state fields not saved in checkpointer
**Solution**: Ensure checkpointer is typed correctly: `InMemoryCheckpointer[YourStateType]()`

## Examples

See the working examples in:
- `/examples/custom-state/custom_state.py` - CV analysis example
- `/test_custom_state_comprehensive.py` - Comprehensive test cases
