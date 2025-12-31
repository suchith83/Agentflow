# Memory Architecture Plan

## Executive Summary

This document outlines the comprehensive architecture for memory management in the Agentflow framework. The current implementation has several gaps that prevent effective memory consumption and AI-driven memory updates. This plan addresses:

1. **Memory Consumption Patterns** - How agents should retrieve and use memory
2. **AI-Driven Memory Updates** - How agents should automatically update memory
3. **Memory Integration with Graph** - Seamless integration with the graph execution flow
4. **Testability** - Making memory operations mockable and testable

---

## Current State Analysis

### What Exists

| Component | File | Status |
|-----------|------|--------|
| `BaseStore` | `agentflow/store/base_store.py` | ✅ Abstract interface defined |
| `Mem0Store` | `agentflow/store/mem0_store.py` | ✅ Mem0 integration exists |
| `QdrantStore` | `agentflow/store/qdrant_store.py` | ✅ Vector store implementation |
| `MemorySearchResult` | `agentflow/store/store_schema.py` | ✅ Data models defined |
| `BaseEmbedding` | `agentflow/store/embedding/base_embedding.py` | ✅ Embedding abstraction |

### Critical Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| No automatic memory injection | Agents must manually call store | 🔴 High |
| No AI-driven memory update pattern | Memory updates are manual | 🔴 High |
| Store not injected via DI | Hard to mock/test | 🔴 High |
| No memory middleware/hook | Can't intercept messages for memory | 🟡 Medium |
| No memory summarization | Context grows unbounded | 🟡 Medium |
| No memory relevance scoring in response | LLM can't judge memory quality | 🟠 Low |

---

## Proposed Architecture

### 1. Memory Manager Service

Create a new `MemoryManager` that orchestrates memory operations:

```python
# agentflow/memory/memory_manager.py
from typing import Any
from injectq import Inject
from agentflow.store import BaseStore
from agentflow.state import AgentState, Message

class MemoryManager:
    """Orchestrates memory retrieval and storage for agents."""
    
    def __init__(
        self,
        store: BaseStore = Inject[BaseStore],
        auto_retrieve: bool = True,
        auto_store: bool = True,
        relevance_threshold: float = 0.7,
        max_memories: int = 5,
    ):
        self.store = store
        self.auto_retrieve = auto_retrieve
        self.auto_store = auto_store
        self.relevance_threshold = relevance_threshold
        self.max_memories = max_memories
    
    async def retrieve_context(
        self,
        config: dict[str, Any],
        query: str,
        memory_types: list[MemoryType] | None = None,
    ) -> list[MemorySearchResult]:
        """Retrieve relevant memories for the current context."""
        if not self.store:
            return []
        
        results = await self.store.asearch(
            config=config,
            query=query,
            limit=self.max_memories,
            score_threshold=self.relevance_threshold,
            memory_type=memory_types[0] if memory_types and len(memory_types) == 1 else None,
        )
        return results
    
    async def store_interaction(
        self,
        config: dict[str, Any],
        user_message: Message,
        assistant_message: Message,
        memory_type: MemoryType = MemoryType.EPISODIC,
    ) -> str:
        """Store a user-assistant interaction as memory."""
        if not self.store:
            return ""
        
        # Format interaction for storage
        content = f"User: {user_message.text()}\nAssistant: {assistant_message.text()}"
        
        return await self.store.astore(
            config=config,
            content=content,
            memory_type=memory_type,
            metadata={
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_message.id,
            }
        )
    
    async def extract_and_store_facts(
        self,
        config: dict[str, Any],
        message: Message,
        facts: list[str],
    ) -> list[str]:
        """Extract and store semantic facts from a message."""
        if not self.store:
            return []
        
        memory_ids = []
        for fact in facts:
            mem_id = await self.store.astore(
                config=config,
                content=fact,
                memory_type=MemoryType.SEMANTIC,
                metadata={"source_message_id": message.id},
            )
            memory_ids.append(mem_id)
        return memory_ids
```

### 2. Memory-Aware Agent Mixin

Create a mixin that adds memory capabilities to any agent:

```python
# agentflow/memory/memory_mixin.py
from typing import Any, Protocol
from injectq import Inject
from agentflow.state import AgentState, Message

class MemoryAwareMixin:
    """Mixin that adds memory capabilities to agents."""
    
    memory_manager: MemoryManager = Inject[MemoryManager]
    
    async def with_memory_context(
        self,
        state: AgentState,
        config: dict[str, Any],
    ) -> tuple[AgentState, list[MemorySearchResult]]:
        """Enhance state with relevant memory context."""
        if not state.context:
            return state, []
        
        # Get the latest user message for context
        user_messages = [m for m in state.context if m.role == "user"]
        if not user_messages:
            return state, []
        
        query = user_messages[-1].text()
        memories = await self.memory_manager.retrieve_context(config, query)
        
        # Inject memory context into system prompt or state
        if memories:
            memory_context = self._format_memories(memories)
            # Add to state's context_summary or metadata
            state = state.model_copy(update={
                "context_summary": memory_context
            })
        
        return state, memories
    
    def _format_memories(self, memories: list[MemorySearchResult]) -> str:
        """Format memories for inclusion in context."""
        if not memories:
            return ""
        
        lines = ["Relevant memories from past interactions:"]
        for i, mem in enumerate(memories, 1):
            lines.append(f"{i}. [{mem.memory_type.value}] {mem.content}")
        return "\n".join(lines)
    
    async def store_interaction_memory(
        self,
        state: AgentState,
        config: dict[str, Any],
        user_msg: Message,
        assistant_msg: Message,
    ) -> None:
        """Store the interaction in memory."""
        await self.memory_manager.store_interaction(
            config=config,
            user_message=user_msg,
            assistant_message=assistant_msg,
        )
```

### 3. Memory Callback Hooks

Integrate memory operations into the callback system:

```python
# agentflow/memory/memory_callbacks.py
from agentflow.utils.callbacks import Callback

class MemoryRetrievalCallback(Callback):
    """Callback that retrieves relevant memories before node execution."""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    async def on_node_start(
        self,
        node_name: str,
        state: AgentState,
        config: dict[str, Any],
    ) -> AgentState:
        """Retrieve and inject memories before node runs."""
        if not state.context:
            return state
        
        user_messages = [m for m in state.context if m.role == "user"]
        if not user_messages:
            return state
        
        query = user_messages[-1].text()
        memories = await self.memory_manager.retrieve_context(config, query)
        
        if memories:
            memory_summary = self._format_memories(memories)
            state = state.model_copy(update={
                "context_summary": f"{state.context_summary or ''}\n{memory_summary}".strip()
            })
        
        return state


class MemoryStorageCallback(Callback):
    """Callback that stores interactions after node execution."""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        store_nodes: list[str] | None = None,
    ):
        self.memory_manager = memory_manager
        self.store_nodes = store_nodes or ["MAIN"]
    
    async def on_node_end(
        self,
        node_name: str,
        state: AgentState,
        config: dict[str, Any],
        result: Any,
    ) -> None:
        """Store interaction in memory after node completes."""
        if node_name not in self.store_nodes:
            return
        
        if len(state.context) < 2:
            return
        
        # Find last user-assistant pair
        user_msg = None
        assistant_msg = None
        
        for msg in reversed(state.context):
            if msg.role == "assistant" and assistant_msg is None:
                assistant_msg = msg
            elif msg.role == "user" and user_msg is None:
                user_msg = msg
            
            if user_msg and assistant_msg:
                break
        
        if user_msg and assistant_msg:
            await self.memory_manager.store_interaction(
                config=config,
                user_message=user_msg,
                assistant_message=assistant_msg,
            )
```

### 4. DI Registration Pattern

Register memory components via InjectQ for easy mocking:

```python
# agentflow/memory/__init__.py
from injectq import InjectQ

def configure_memory(
    container: InjectQ,
    store: BaseStore,
    auto_retrieve: bool = True,
    auto_store: bool = True,
    relevance_threshold: float = 0.7,
    max_memories: int = 5,
) -> None:
    """Configure memory services in the DI container."""
    # Register the store
    container.bind_instance(BaseStore, store, allow_concrete=True)
    
    # Create and register memory manager
    memory_manager = MemoryManager(
        store=store,
        auto_retrieve=auto_retrieve,
        auto_store=auto_store,
        relevance_threshold=relevance_threshold,
        max_memories=max_memories,
    )
    container.bind_instance(MemoryManager, memory_manager, allow_concrete=True)
```

---

## Usage Examples

### Example 1: Basic Memory-Enabled Agent

```python
from agentflow.graph import StateGraph
from agentflow.memory import MemoryManager, configure_memory, MemoryRetrievalCallback
from agentflow.store import QdrantStore
from agentflow.store.embedding import OpenAIEmbedding
from injectq import InjectQ

# Setup
container = InjectQ()
store = QdrantStore(embedding=OpenAIEmbedding())
configure_memory(container, store)

# Create graph with memory
graph = StateGraph(container=container)

# Memory is automatically available via DI
async def memory_aware_agent(
    state: AgentState,
    config: dict[str, Any],
    memory_manager: MemoryManager = Inject[MemoryManager],
):
    # Retrieve relevant memories
    memories = await memory_manager.retrieve_context(
        config, 
        state.context[-1].text()
    )
    
    # Use memories in LLM call
    memory_context = "\n".join([m.content for m in memories])
    
    messages = convert_messages(
        system_prompts=[{
            "role": "system",
            "content": f"You are helpful. Context from memory:\n{memory_context}"
        }],
        state=state
    )
    
    response = await acompletion(model="gpt-4", messages=messages)
    result = ModelResponseConverter(response, converter="litellm")
    
    # Store the interaction
    await memory_manager.store_interaction(
        config,
        state.context[-1],  # user message
        result.message,     # assistant message
    )
    
    return result

graph.add_node("MAIN", memory_aware_agent)
graph.set_entry_point("MAIN")
graph.add_edge("MAIN", END)

compiled = graph.compile()
```

### Example 2: Using Memory Callbacks (Zero-Code Memory)

```python
from agentflow.memory import (
    configure_memory,
    MemoryRetrievalCallback,
    MemoryStorageCallback,
)

# Setup
container = InjectQ()
store = QdrantStore(embedding=OpenAIEmbedding())
configure_memory(container, store)

# Get memory manager for callbacks
memory_manager = container.resolve(MemoryManager)

# Create callbacks
callbacks = CallbackManager([
    MemoryRetrievalCallback(memory_manager),
    MemoryStorageCallback(memory_manager, store_nodes=["MAIN"]),
])

# Regular agent - no memory code needed!
async def simple_agent(state: AgentState, config: dict):
    # Memory is injected via callback - check context_summary
    system_prompt = f"You are helpful. {state.context_summary or ''}"
    
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": system_prompt}],
        state=state
    )
    response = await acompletion(model="gpt-4", messages=messages)
    return ModelResponseConverter(response)

graph = StateGraph(container=container)
graph.add_node("MAIN", simple_agent)
graph.set_entry_point("MAIN")
graph.add_edge("MAIN", END)

# Memory callbacks handle everything
compiled = graph.compile(callback_manager=callbacks)
```

### Example 3: AI-Driven Fact Extraction

```python
from agentflow.memory import MemoryManager

FACT_EXTRACTION_PROMPT = """
Extract key facts from this conversation that should be remembered.
Return a JSON array of strings, each being a distinct fact.

Conversation:
{conversation}

Facts (JSON array):
"""

async def extract_facts_agent(
    state: AgentState,
    config: dict[str, Any],
    memory_manager: MemoryManager = Inject[MemoryManager],
):
    # Get last exchange
    if len(state.context) < 2:
        return state
    
    conversation = "\n".join([
        f"{m.role}: {m.text()}" 
        for m in state.context[-4:]  # Last 2 exchanges
    ])
    
    # Ask LLM to extract facts
    response = await acompletion(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": FACT_EXTRACTION_PROMPT.format(conversation=conversation)
        }],
        response_format={"type": "json_object"}
    )
    
    facts = json.loads(response.choices[0].message.content)
    
    # Store extracted facts as semantic memories
    await memory_manager.extract_and_store_facts(
        config=config,
        message=state.context[-1],
        facts=facts,
    )
    
    return state
```

---

## Testing Strategy

### 1. Mock Store for Unit Tests

```python
# tests/memory/conftest.py
import pytest
from agentflow.store import BaseStore
from agentflow.store.store_schema import MemorySearchResult, MemoryType

class MockStore(BaseStore):
    """In-memory mock store for testing."""
    
    def __init__(self):
        self.memories: dict[str, MemorySearchResult] = {}
        self._search_results: list[MemorySearchResult] = []
    
    def set_search_results(self, results: list[MemorySearchResult]):
        """Pre-configure search results for testing."""
        self._search_results = results
    
    async def astore(self, config, content, **kwargs) -> str:
        mem_id = str(uuid4())
        self.memories[mem_id] = MemorySearchResult(
            id=mem_id,
            content=content,
            score=1.0,
            memory_type=kwargs.get("memory_type", MemoryType.EPISODIC),
        )
        return mem_id
    
    async def asearch(self, config, query, **kwargs) -> list[MemorySearchResult]:
        return self._search_results
    
    async def aget(self, config, memory_id, **kwargs):
        return self.memories.get(memory_id)
    
    async def aget_all(self, config, **kwargs):
        return list(self.memories.values())
    
    async def aupdate(self, config, memory_id, **kwargs):
        if memory_id in self.memories:
            if "content" in kwargs:
                self.memories[memory_id].content = kwargs["content"]
            return True
        return False
    
    async def adelete(self, config, memory_id, **kwargs):
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False

@pytest.fixture
def mock_store():
    return MockStore()

@pytest.fixture
def container_with_memory(mock_store):
    container = InjectQ()
    configure_memory(container, mock_store)
    return container
```

### 2. Memory Manager Tests

```python
# tests/memory/test_memory_manager.py
import pytest
from agentflow.memory import MemoryManager
from agentflow.state import Message

class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_retrieve_context(self, mock_store):
        # Arrange
        mock_store.set_search_results([
            MemorySearchResult(
                id="1",
                content="User likes pizza",
                score=0.9,
                memory_type=MemoryType.SEMANTIC,
            )
        ])
        
        manager = MemoryManager(store=mock_store)
        config = {"user_id": "test", "thread_id": "t1"}
        
        # Act
        results = await manager.retrieve_context(config, "What food do I like?")
        
        # Assert
        assert len(results) == 1
        assert results[0].content == "User likes pizza"
    
    @pytest.mark.asyncio
    async def test_store_interaction(self, mock_store):
        manager = MemoryManager(store=mock_store)
        config = {"user_id": "test", "thread_id": "t1"}
        
        user_msg = Message.text_message("Hello", role="user")
        assistant_msg = Message.text_message("Hi there!", role="assistant")
        
        # Act
        mem_id = await manager.store_interaction(
            config, user_msg, assistant_msg
        )
        
        # Assert
        assert mem_id in mock_store.memories
        assert "Hello" in mock_store.memories[mem_id].content
        assert "Hi there!" in mock_store.memories[mem_id].content
```

### 3. Integration Test with Graph

```python
# tests/memory/test_memory_integration.py
import pytest
from agentflow.graph import StateGraph
from agentflow.memory import configure_memory, MemoryManager
from agentflow.state import AgentState, Message
from injectq import InjectQ, Inject

class TestMemoryGraphIntegration:
    @pytest.mark.asyncio
    async def test_memory_injection_in_node(self, mock_store):
        # Setup DI
        container = InjectQ()
        configure_memory(container, mock_store)
        
        # Pre-populate memories
        mock_store.set_search_results([
            MemorySearchResult(
                id="1",
                content="User's name is John",
                score=0.95,
            )
        ])
        
        # Node that uses memory
        async def memory_node(
            state: AgentState,
            config: dict,
            memory_manager: MemoryManager = Inject[MemoryManager],
        ):
            memories = await memory_manager.retrieve_context(
                config, 
                state.context[-1].text()
            )
            # Return memory content for verification
            return [Message.text_message(
                f"Found {len(memories)} memories: {memories[0].content}",
                role="assistant"
            )]
        
        # Build graph
        graph = StateGraph(container=container)
        graph.add_node("MAIN", memory_node)
        graph.set_entry_point("MAIN")
        graph.add_edge("MAIN", END)
        
        compiled = graph.compile()
        
        # Execute
        result = await compiled.ainvoke({
            "messages": [Message.text_message("What's my name?", role="user")]
        }, config={"user_id": "test", "thread_id": "t1"})
        
        # Verify
        assert "Found 1 memories" in result["messages"][-1].content
        assert "John" in result["messages"][-1].content
```

---

## File Structure

```
agentflow/
├── memory/
│   ├── __init__.py           # configure_memory() and exports
│   ├── memory_manager.py     # MemoryManager class
│   ├── memory_mixin.py       # MemoryAwareMixin
│   ├── memory_callbacks.py   # MemoryRetrievalCallback, MemoryStorageCallback
│   └── extractors/
│       ├── __init__.py
│       ├── base_extractor.py # Abstract fact extractor
│       └── llm_extractor.py  # LLM-based fact extraction
├── store/
│   ├── ... (existing)
│   └── mock_store.py         # MockStore for testing (NEW)
tests/
├── memory/
│   ├── __init__.py
│   ├── conftest.py           # Fixtures
│   ├── test_memory_manager.py
│   ├── test_memory_callbacks.py
│   └── test_memory_integration.py
```

---

## Migration Path

### Phase 1: Core Memory Manager (Week 1)
- [ ] Create `agentflow/memory/` module
- [ ] Implement `MemoryManager` class
- [ ] Add DI configuration helper
- [ ] Create `MockStore` for testing

### Phase 2: Callback Integration (Week 2)
- [ ] Implement `MemoryRetrievalCallback`
- [ ] Implement `MemoryStorageCallback`
- [ ] Update `Agent` class to support memory mixin
- [ ] Add examples

### Phase 3: Advanced Features (Week 3)
- [ ] LLM-based fact extraction
- [ ] Memory summarization
- [ ] Memory decay/forgetting strategies
- [ ] Cross-session memory consolidation

---

## Open Questions

1. **Memory Scope**: Should memories be scoped to `user_id`, `thread_id`, or both?
   - Recommendation: `user_id` for long-term, `thread_id` for session-specific

2. **Memory Deduplication**: How to handle duplicate or near-duplicate memories?
   - Recommendation: Vector similarity threshold for dedup before storage

3. **Memory Limits**: Should there be hard limits on memories per user?
   - Recommendation: Configurable limits with LRU-style eviction

4. **Privacy**: How to handle memory deletion/GDPR compliance?
   - Recommendation: Add `forget_user()` and `forget_memory()` methods

---

## Summary

This plan transforms the current passive store into an active memory system that:

1. ✅ **Auto-retrieves** relevant context before LLM calls
2. ✅ **Auto-stores** interactions after responses  
3. ✅ **Injects via DI** for easy testing/mocking
4. ✅ **Uses callbacks** for zero-code memory integration
5. ✅ **Extracts facts** via AI for semantic memory building
6. ✅ **Fully testable** with mock stores and DI overrides
