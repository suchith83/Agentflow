# Agentflow Architecture Plans

This folder contains comprehensive architecture plans for improving the Agentflow framework.

## 📋 Documents

| Document | Description | Priority |
|----------|-------------|----------|
| [MemoryPlan.md](MemoryPlan.md) | Memory architecture, consumption patterns, AI-driven updates | 🔴 Critical |
| [AgentRegistrationPlan.md](AgentRegistrationPlan.md) | Testable agents via BaseAgent/TestAgent pattern, override_node() | 🔴 Critical |
| [ArchitectureGaps.md](ArchitectureGaps.md) | Comprehensive gap analysis across all areas | 📊 Overview |
| [TestingPlan.md](TestingPlan.md) | User requirements for simple testing approach | 📝 Requirements |

---

## 🎯 Key Problems Addressed

### 1. Memory Not Properly Consumable
**Problem**: Store implementations exist but there's no pattern for:
- How agents retrieve relevant memories
- How AI automatically updates memories
- How to test memory operations

**Solution**: See [MemoryPlan.md](MemoryPlan.md)
- `MemoryManager` service for orchestration
- `MemoryRetrievalCallback` for automatic context injection
- `MemoryStorageCallback` for automatic interaction storage
- `InMemoryStore` for testing (like InMemoryCheckpointer)

### 2. Agent Registration Not Testable
**Problem**: The current pattern makes testing difficult:
```python
agent = Agent(model="gpt-4", ...)
graph.add_node("MAIN", agent)  # Can't swap for tests
```

**Solution**: See [AgentRegistrationPlan.md](AgentRegistrationPlan.md)
- **BaseAgent pattern** - `Agent` and `TestAgent` inherit from `BaseAgent`
- **override_node()** - `graph.override_node("MAIN", test_func)` for easy swaps
- **TestContext** - Optional helper for test setup
- **InMemoryStore** - Like InMemoryCheckpointer, no embeddings needed

---

## 🧪 Testing Philosophy

**Simple for prototyping, powerful for production.**

```python
# Approach 1: Use TestAgent (for Agent class testing)
from agentflow.testing import TestAgent

test_agent = TestAgent(responses=["Mock response"])
graph.add_node("MAIN", test_agent)

# Approach 2: Use override_node() (for any node)
graph.override_node("MAIN", my_test_function)

# Approach 3: Use TestContext (for full isolation)
with TestContext() as ctx:
    graph = ctx.create_graph()
    ...
```

---

## 🗓️ Implementation Roadmap

### Week 1-2: Core Testability
- [ ] Create `BaseAgent` abstract class
- [ ] Modify `Agent` to inherit from `BaseAgent`
- [ ] Create `TestAgent` for testing
- [ ] Add `override_node()` to StateGraph
- [ ] Create `agentflow/testing/` module

### Week 3-4: Memory System
- [ ] Create `InMemoryStore` (like InMemoryCheckpointer)
- [ ] Implement `MemoryManager`
- [ ] Implement memory callbacks
- [ ] Test memory integration

### Week 5-6: Production Features
- [ ] Add retry/fallback mechanism
- [ ] Implement request tracing
- [ ] Add context window management

### Week 7-8: Developer Experience
- [ ] Graph serialization (to_yaml, from_yaml)
- [ ] Graph visualization (to_mermaid)
- [ ] Subgraph support

---

## 🏗️ Proposed Module Structure

```
agentflow/
├── graph/
│   ├── base_agent.py         # NEW: BaseAgent abstract class
│   ├── agent.py              # MODIFIED: Inherits from BaseAgent
│   ├── state_graph.py        # MODIFIED: Add override_node()
│   └── ...
│
├── testing/                  # NEW: Test utilities
│   ├── __init__.py           # TestContext, exports
│   └── test_agent.py         # TestAgent implementation
│
├── store/
│   ├── in_memory_store.py    # NEW: Test store (like InMemoryCheckpointer)
│   └── ...
│
├── memory/                   # NEW: Memory orchestration
│   ├── __init__.py
│   ├── memory_manager.py     # Core manager
│   └── memory_callbacks.py   # Auto memory hooks
```

---

## 📊 Gap Summary

From [ArchitectureGaps.md](ArchitectureGaps.md):

| Severity | Count | Examples |
|----------|-------|----------|
| 🔴 Critical | 7 | LLM abstraction, memory injection, test fixtures |
| 🟡 Medium | 7 | Streaming accumulator, context management, subgraphs |
| 🟠 Low | 6 | Rate limiting, metrics, warm-up |

---

## ✅ Quick Wins

These can be implemented immediately:

1. **Better error messages** - Add context to exceptions
2. **Graph visualization** - `to_mermaid()` method
3. **Config validation** - Pydantic `GraphConfig` model
4. **MockStore** - In-memory store for testing

---

## 🔗 Related Files

- Main code: `agentflow/`
- Tests: `tests/`
- Examples: `examples/`
- Current docs: `docs/`
