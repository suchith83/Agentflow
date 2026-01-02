# Agent Registration & Testability Plan (Revised v2)

## Executive Summary

This document provides the **simplest practical approach** to making Agentflow testable while keeping it simple for beginners and powerful for production. 

**Core Principles:**
1. **Simple for prototyping** - No forced abstractions, plain functions work
2. **Powerful for production** - Full testability when needed
3. **BaseAgent pattern** - Inheritance for Agent class (Agent, TestAgent)
4. **Override nodes** - `graph.override_node("NAME", test_func)` for functions
5. **Keep Node/Edge internal** - No need to mock them, they're not exposed

---

## Current State Analysis

### What Works Well ✅

```python
# This already works - node functions can be simple
async def my_node(state: AgentState, config: dict):
    return [Message.text_message("Hello")]

graph = StateGraph()
graph.add_node("MAIN", my_node)  # ✅ Simple and direct
```

### What's Hard to Test ❌

```python
# The Agent class calls litellm.acompletion() directly
agent = Agent(model="gpt-4", system_prompt=[...])
graph.add_node("MAIN", agent)
# ❌ Can't swap to a test agent easily
```

---

## Solution Overview

### Two Approaches for Two Use Cases

| Use Case | Solution | Complexity |
|----------|----------|------------|
| Testing Agent class | `TestAgent` inherits from `BaseAgent` | Simple inheritance |
| Testing node functions | `graph.override_node()` | One-line swap |

---

## Part 1: Agent Class Testability

### 1.1 Create BaseAgent Abstract Class

```python
# agentflow/graph/base_agent.py
from abc import ABC, abstractmethod
from typing import Any

from agentflow.state import AgentState


class BaseAgent(ABC):
    """Base class for all agents - production and test.
    
    Provides the common interface that all agents must implement.
    This allows swapping between Agent and TestAgent seamlessly.
    """
    
    def __init__(
        self,
        model: str,
        system_prompt: list[dict[str, Any]],
        tools: list | None = None,
        **kwargs,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.kwargs = kwargs
    
    @abstractmethod
    async def execute(
        self,
        state: AgentState,
        config: dict[str, Any],
    ):
        """Execute the agent logic.
        
        Args:
            state: Current agent state with context
            config: Execution configuration
            
        Returns:
            ModelResponseConverter, list[Message], or dict
        """
        pass
    
    @abstractmethod
    async def _call_llm(
        self,
        messages: list[dict],
        tools: list | None = None,
        **kwargs,
    ):
        """Make the actual LLM call.
        
        This is the method that differs between production and test.
        """
        pass
```

### 1.2 Modify Agent to Inherit from BaseAgent

```python
# agentflow/graph/agent.py
from agentflow.graph.base_agent import BaseAgent

class Agent(BaseAgent):
    """Production agent that calls real LLM via litellm."""
    
    def __init__(
        self,
        model: str,
        system_prompt: list[dict[str, Any]],
        tools: list[Callable] | ToolNode | None = None,
        **llm_kwargs,
    ):
        super().__init__(model, system_prompt, tools, **llm_kwargs)
        
        if not HAS_LITELLM:
            raise ImportError(
                "litellm is required for Agent class. "
                "Install it with: pip install 10xscale-agentflow[litellm]"
            )
        
        self._tool_node = self._setup_tools()
    
    async def _call_llm(
        self,
        messages: list[dict],
        tools: list | None = None,
        **kwargs,
    ):
        """Call real LLM via litellm.acompletion()."""
        from litellm import acompletion
        
        return await acompletion(
            model=self.model,
            messages=messages,
            tools=tools,
            **self.kwargs,
            **kwargs,
        )
    
    async def execute(self, state: AgentState, config: dict[str, Any]):
        # ... existing execute logic, but calls self._call_llm() instead of acompletion()
        messages = convert_messages(state=state, system_prompts=self.system_prompt)
        
        tools = []
        if self._tool_node:
            tools = await self._tool_node.all_tools()
        
        response = await self._call_llm(messages, tools=tools, stream=config.get("is_stream", False))
        
        return ModelResponseConverter(response, converter="litellm")
```

### 1.3 Create TestAgent

```python
# agentflow/testing/test_agent.py
from typing import Any
from agentflow.graph.base_agent import BaseAgent
from agentflow.state import AgentState
from agentflow.state.message import Message


class TestAgent(BaseAgent):
    """Test agent for unit testing - returns predefined responses.
    
    Use this to swap out the production Agent in tests.
    
    Example:
        ```python
        # Production code
        agent = Agent(model="gpt-4", system_prompt=[...])
        
        # Test code
        test_agent = TestAgent(
            model="gpt-4",
            system_prompt=[...],
            responses=["Hello from test!"]
        )
        graph.override_node("MAIN", test_agent)
        ```
    """
    
    def __init__(
        self,
        model: str = "test-model",
        system_prompt: list[dict[str, Any]] | None = None,
        responses: list[str] | None = None,
        tools: list | None = None,
        **kwargs,
    ):
        super().__init__(model, system_prompt or [], tools, **kwargs)
        self.responses = responses or ["Test response"]
        self.call_count = 0
        self.call_history: list[dict] = []
    
    async def _call_llm(
        self,
        messages: list[dict],
        tools: list | None = None,
        **kwargs,
    ):
        """Return predefined response instead of calling LLM."""
        self.call_count += 1
        self.call_history.append({
            "messages": messages,
            "tools": tools,
            "kwargs": kwargs,
        })
        
        # Get next response (cycles through list)
        idx = (self.call_count - 1) % len(self.responses)
        content = self.responses[idx]
        
        # Return dict matching litellm structure
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
    
    async def execute(self, state: AgentState, config: dict[str, Any]):
        """Execute test agent - returns mock response."""
        from agentflow.adapters.llm import ModelResponseConverter
        from agentflow.utils.converter import convert_messages
        
        messages = convert_messages(state=state, system_prompts=self.system_prompt)
        response = await self._call_llm(messages)
        
        return ModelResponseConverter(response, converter="litellm")
    
    # Assertion helpers
    def assert_called(self):
        """Assert the agent was called at least once."""
        assert self.call_count > 0, "TestAgent was never called"
    
    def assert_called_times(self, n: int):
        """Assert the agent was called exactly n times."""
        assert self.call_count == n, f"Expected {n} calls, got {self.call_count}"
    
    def get_last_messages(self) -> list[dict]:
        """Get messages from the last call."""
        if not self.call_history:
            return []
        return self.call_history[-1]["messages"]
```

---

## Part 2: Node Function Testability

### 2.1 Add `override_node()` to StateGraph

```python
# agentflow/graph/state_graph.py

class StateGraph[StateT: AgentState]:
    """..."""
    
    def override_node(
        self,
        name: str,
        func: Union[Callable, "ToolNode", "BaseAgent"],
    ) -> "StateGraph":
        """Override an existing node with a different function.
        
        Use this in tests to swap production nodes with test doubles.
        The node must already exist in the graph.
        
        Args:
            name: Name of the existing node to override
            func: New function, ToolNode, or Agent to use
            
        Returns:
            StateGraph: The graph instance for method chaining.
            
        Raises:
            KeyError: If the node doesn't exist
            
        Example:
            ```python
            # Production
            graph = StateGraph()
            graph.add_node("MAIN", production_agent)
            
            # Test - override with test agent
            graph.override_node("MAIN", test_agent)
            ```
        """
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' does not exist. Use add_node() first.")
        
        # Create new Node with same name but different function
        self.nodes[name] = Node(name, func, self._publisher)
        logger.debug("Overrode node '%s' with new function", name)
        return self
```

### 2.2 Usage Examples

```python
# tests/test_my_workflow.py
import pytest
from agentflow import StateGraph, AgentState, Message
from agentflow.testing import TestAgent

# Production node
async def production_node(state: AgentState, config: dict):
    # This calls real LLM, hits APIs, etc.
    pass

# Test double
async def test_node(state: AgentState, config: dict):
    return [Message.text_message("Test response")]


@pytest.mark.asyncio
async def test_workflow_with_override():
    """Test using override_node() for functions."""
    graph = StateGraph()
    graph.add_node("MAIN", production_node)
    graph.set_entry_point("MAIN")
    graph.add_edge("MAIN", END)
    
    # Override for test
    graph.override_node("MAIN", test_node)
    
    compiled = graph.compile()
    result = await compiled.ainvoke({"messages": [Message.text_message("Hi")]})
    
    assert "Test response" in result["messages"][-1].text()


@pytest.mark.asyncio
async def test_workflow_with_test_agent():
    """Test using TestAgent for Agent nodes."""
    # Create test agent
    test_agent = TestAgent(
        model="gpt-4",
        responses=["I'm a test agent!"]
    )
    
    graph = StateGraph()
    graph.add_node("MAIN", test_agent)
    graph.set_entry_point("MAIN")
    graph.add_edge("MAIN", END)
    
    compiled = graph.compile()
    result = await compiled.ainvoke({"messages": [Message.text_message("Hi")]})
    
    # Assert behavior
    test_agent.assert_called()
    assert "test agent" in result["messages"][-1].text().lower()
```

---

## Part 3: InMemoryStore (Like InMemoryCheckpointer)

```python
# agentflow/store/in_memory_store.py
from uuid import uuid4
from agentflow.store.base_store import BaseStore
from agentflow.store.store_schema import MemorySearchResult, MemoryType


class InMemoryStore(BaseStore):
    """In-memory store for testing - no external dependencies.
    
    Like InMemoryCheckpointer, this provides a simple in-memory
    implementation for testing without requiring databases or embeddings.
    
    Example:
        ```python
        store = InMemoryStore()
        
        # Pre-configure search results
        store.set_search_results([
            MemorySearchResult(id="1", content="Relevant memory", score=0.9)
        ])
        
        graph = StateGraph()
        compiled = graph.compile(store=store)
        ```
    """
    
    def __init__(self):
        self.memories: dict[str, MemorySearchResult] = {}
        self._search_results: list[MemorySearchResult] = []
    
    def set_search_results(self, results: list[MemorySearchResult]):
        """Pre-configure search results for testing."""
        self._search_results = results
    
    async def astore(self, config, content, **kwargs) -> str:
        """Store a memory."""
        mem_id = str(uuid4())
        self.memories[mem_id] = MemorySearchResult(
            id=mem_id,
            content=content if isinstance(content, str) else str(content),
            score=1.0,
            memory_type=kwargs.get("memory_type", MemoryType.EPISODIC),
        )
        return mem_id
    
    async def asearch(self, config, query, **kwargs) -> list[MemorySearchResult]:
        """Search memories - returns pre-configured results or text match."""
        if self._search_results:
            return self._search_results
        
        # Simple text search fallback
        results = []
        query_lower = query.lower() if isinstance(query, str) else ""
        for mem in self.memories.values():
            if query_lower in mem.content.lower():
                results.append(mem)
        return results[:kwargs.get("limit", 10)]
    
    async def aget(self, config, memory_id, **kwargs):
        """Get a specific memory."""
        return self.memories.get(memory_id)
    
    async def aget_all(self, config, **kwargs) -> list[MemorySearchResult]:
        """Get all memories."""
        return list(self.memories.values())
    
    async def aupdate(self, config, memory_id, **kwargs) -> bool:
        """Update a memory."""
        if memory_id in self.memories and "content" in kwargs:
            self.memories[memory_id].content = kwargs["content"]
            return True
        return False
    
    async def adelete(self, config, memory_id, **kwargs) -> bool:
        """Delete a memory."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False
    
    def clear(self):
        """Clear all memories and pre-configured results."""
        self.memories.clear()
        self._search_results.clear()
```

---

## Part 4: Optional TestContext Helper

**Note:** This is optional - users can test without it. Provided for convenience.

```python
# agentflow/testing/__init__.py
from agentflow.testing.test_agent import TestAgent
from agentflow.store.in_memory_store import InMemoryStore


class TestContext:
    """Optional helper for test setup.
    
    Provides convenience methods for common test patterns.
    Users don't need to use this - they can set up tests manually.
    
    Example:
        ```python
        with TestContext() as ctx:
            graph = ctx.create_graph()
            graph.add_node("MAIN", ctx.create_test_agent(responses=["Hi!"]))
            # ... test
        ```
    """
    
    def __init__(self):
        from injectq import InjectQ
        self.container = InjectQ()
        self.store = InMemoryStore()
    
    def __enter__(self):
        self.container.activate()
        return self
    
    def __exit__(self, *args):
        self.container.deactivate()
    
    def create_graph(self, state=None):
        """Create a graph with test container."""
        from agentflow import StateGraph
        return StateGraph(state=state, container=self.container)
    
    def create_test_agent(
        self,
        responses: list[str] | None = None,
        model: str = "test-model",
    ) -> TestAgent:
        """Create a TestAgent with predefined responses."""
        return TestAgent(model=model, responses=responses)
    
    def get_store(self) -> InMemoryStore:
        """Get the in-memory store for this test context."""
        return self.store


# Pytest fixture for convenience
def pytest_fixture():
    """
    # In conftest.py:
    from agentflow.testing import TestContext
    
    @pytest.fixture
    def test_ctx():
        with TestContext() as ctx:
            yield ctx
    """
    pass
```

---

## Part 5: InvokeHandler Node Resolution (Advanced)

**Question from TestingPlan.md:** *"How can InvokeHandler get nodes from InjectQ instead of graph nodes list?"*

### Current Behavior

```python
# agentflow/graph/compiled_graph.py
self._invoke_handler = InvokeHandler[StateT](
    nodes=state_graph.nodes,  # Dict passed directly
    edges=state_graph.edges,
    ...
)
```

### Recommendation: Keep Current Approach

The current approach is **correct** for these reasons:

1. **Nodes dict IS the graph** - Traversal needs the full dict for routing
2. **InjectQ is for services** - Not for graph structure
3. **override_node() modifies the dict** - So tests work naturally

**However**, we already register factories for node lookup:

```python
# Already in state_graph.py compile()
self._container.bind_factory("get_node", lambda x: self.nodes[x])
```

If needed, `InvokeHandler` could use this factory, but it's unnecessary since the dict reference stays valid after `override_node()`.

### Why This Works

```python
graph = StateGraph()
graph.add_node("MAIN", production_agent)  # self.nodes["MAIN"] = Node(...)

# Later in test:
graph.override_node("MAIN", test_agent)  # self.nodes["MAIN"] = Node(test_agent)

# InvokeHandler already has reference to self.nodes dict
# When it does self.nodes["MAIN"], it gets the NEW test node!
```

The `nodes` dict is passed by reference, so overrides are reflected automatically.

---

## Part 6: Testing ToolNode

### Question: Should we write tests for ToolNode?

**Answer: Yes, but test by mocking the underlying tool sources, not ToolNode itself.**

### Understanding ToolNode Architecture

ToolNode aggregates tools from multiple sources:
1. **Local functions** - Plain Python callables
2. **MCP tools** - Model Context Protocol (FastMCP) tools via client
3. **Composio tools** - External integrations via ComposioAdapter
4. **LangChain tools** - LangChain tool ecosystem
5. **Remote tools** - Frontend/external tool registrations

### Testing Strategy: Mock at the Source Level

**Don't create TestToolNode - instead mock the tool sources:**

#### Approach 1: Local Tools - Use Mock Functions (Simplest)

```python
# Production tools
def get_weather(city: str) -> str:
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()

# Test mock
def mock_get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 72°F"

# Test
@pytest.mark.asyncio
async def test_with_local_tools():
    # Just pass mock functions to ToolNode
    test_tools = ToolNode([mock_get_weather])
    
    graph = StateGraph()
    graph.add_node("TOOLS", test_tools)
    # ...
```

#### Approach 2: MCP Tools - Mock the MCP Client

```python
# Create a mock MCP client
class MockMCPClient:
    async def list_tools(self):
        return [
            {"name": "mcp_weather", "description": "Get weather", "inputSchema": {...}}
        ]
    
    async def call_tool(self, name: str, arguments: dict):
        if name == "mcp_weather":
            return {"content": [{"type": "text", "text": "Mock MCP weather: Sunny"}]}
        raise ValueError(f"Unknown tool: {name}")

# Test
@pytest.mark.asyncio
async def test_with_mcp_tools():
    mock_client = MockMCPClient()
    tools = ToolNode([], client=mock_client)
    
    # Tools will use the mock client
    all_tools = await tools.all_tools()
    assert "mcp_weather" in [t["function"]["name"] for t in all_tools]
```

#### Approach 3: Composio Tools - Mock ComposioAdapter

```python
class MockComposioAdapter:
    def list_raw_tools_for_llm(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "composio_github_create_issue",
                    "description": "Create GitHub issue",
                    "parameters": {...}
                }
            }
        ]
    
    def execute(self, slug: str, arguments: dict, **kwargs):
        return {
            "successful": True,
            "data": {"issue_number": 123, "url": "https://github.com/..."}
        }

# Test
@pytest.mark.asyncio
async def test_with_composio_tools():
    mock_composio = MockComposioAdapter()
    tools = ToolNode([], composio_adapter=mock_composio)
    
    result = await tools.invoke(
        name="composio_github_create_issue",
        args={"title": "Test issue"},
        tool_call_id="call_123",
        config={},
        state=AgentState(),
    )
    assert "issue_number" in result.text()
```

#### Approach 4: Mix Multiple Sources (Realistic Test)

```python
@pytest.mark.asyncio
async def test_multi_source_tools():
    # Mock local tool
    def mock_calculator(a: int, b: int) -> int:
        return a + b
    
    # Mock MCP client
    mock_mcp = MockMCPClient()
    
    # Mock Composio
    mock_composio = MockComposioAdapter()
    
    # Create ToolNode with all sources
    tools = ToolNode(
        [mock_calculator],
        client=mock_mcp,
        composio_adapter=mock_composio,
    )
    
    # All tools available
    all_tools = await tools.all_tools()
    tool_names = [t["function"]["name"] for t in all_tools]
    
    assert "mock_calculator" in tool_names
    assert "mcp_weather" in tool_names
    assert "composio_github_create_issue" in tool_names
```

#### Approach 5: Spy Pattern for Call Tracking

```python
class SpyWeatherTool:
    def __init__(self):
        self.calls: list[dict] = []
    
    def __call__(self, city: str, units: str = "metric") -> str:
        self.calls.append({"city": city, "units": units})
        return f"Mock weather for {city} in {units}"

# Test
@pytest.mark.asyncio
async def test_tool_call_tracking():
    spy_weather = SpyWeatherTool()
    tools = ToolNode([spy_weather])
    
    await tools.invoke(
        name="spy_weather",
        args={"city": "Tokyo", "units": "celsius"},
        tool_call_id="call_123",
        config={},
        state=AgentState(),
    )
    
    # Verify call was made with correct args
    assert len(spy_weather.calls) == 1
    assert spy_weather.calls[0]["city"] == "Tokyo"
    assert spy_weather.calls[0]["units"] == "celsius"
```

### Testing ToolNode in Graph Context

```python
@pytest.mark.asyncio
async def test_agent_with_tool_node():
    # Mock tools
    def mock_search(query: str) -> str:
        return f"Mock search results for: {query}"
    
    # Create mock ToolNode
    mock_tools = ToolNode([mock_search])
    
    # Create test agent that will "call" tools
    test_agent = TestAgent(
        model="gpt-4",
        responses=[
            "I need to search for that",  # First response with tool call
            "Based on the search results, here's the answer"  # Final response
        ]
    )
    
    # Build graph
    graph = StateGraph()
    graph.add_node("AGENT", test_agent)
    graph.add_node("TOOLS", mock_tools)
    
    # Setup routing logic
    def route_after_agent(state: AgentState) -> str:
        last_msg = state.context[-1]
        # If agent wants to call tools, go to TOOLS
        if last_msg.has_tool_calls():
            return "TOOLS"
        return END
    
    graph.set_entry_point("AGENT")
    graph.add_conditional_edges("AGENT", route_after_agent, {"TOOLS": "TOOLS", END: END})
    graph.add_edge("TOOLS", "AGENT")  # Return to agent after tools
    
    compiled = graph.compile()
    result = await compiled.ainvoke({"messages": [Message.text_message("Search for Python")]})
    
    # Verify flow
    assert "Mock search results" in str(result["messages"])
```

**Recommendation:** For most tests, use Approach 1 (mock local functions). For integration tests with MCP/Composio, use Approaches 2-4.

---

## Part 7: Testing Strategy - Override After Compile

### Question: Should we override before or after compile?

**Answer: Support BOTH patterns.**

### Pattern 1: Override Before Compile (For Building Test Graphs)

```python
@pytest.mark.asyncio
async def test_override_before_compile():
    """When you control graph creation, swap nodes before compile."""
    graph = StateGraph()
    graph.add_node("MAIN", production_agent)
    
    # Override BEFORE compile
    graph.override_node("MAIN", test_agent)
    
    compiled = graph.compile()
    result = await compiled.ainvoke(...)
```

**Use this when:** You're creating the graph in your test.

### Pattern 2: Override After Compile (For Pre-Built Graphs)

**You're absolutely right!** If you have a factory that returns a compiled graph, you need to override after compile.

**Important:** This requires handlers to read nodes from InjectQ instead of using cached dict reference.

```python
# Add this method to CompiledGraph
class CompiledGraph:
    def override_node(
        self,
        name: str,
        func: Union[Callable, "ToolNode", "BaseAgent"],
    ) -> "CompiledGraph":
        """Override a node in an already-compiled graph.
        
        Useful for testing pre-built production graphs.
        
        Example:
            ```python
            # Production factory
            def create_production_workflow():
                graph = StateGraph()
                graph.add_node("MAIN", production_agent)
                # ... complex setup
                return graph.compile()
            
            # Test
            compiled = create_production_workflow()
            compiled.override_node("MAIN", test_agent)  # ✅ Override after compile
            result = await compiled.ainvoke(...)
            ```
        """
        if name not in self._state_graph.nodes:
            raise KeyError(f"Node '{name}' does not exist")
        
        # Create new Node and update the graph's dict
        from .node import Node
        new_node = Node(name, func, self._publisher)
        self._state_graph.nodes[name] = new_node
        
        # IMPORTANT: Re-register in InjectQ so handlers pick up the change
        # The handlers read from InjectQ's "get_node" factory
        # This updates the factory's closure to return the new node
        self._state_graph._container.bind_factory(
            "get_node",
            lambda x: self._state_graph.nodes[x]
        )
        
        logger.debug("Overrode node '%s' in compiled graph and updated InjectQ", name)
        return self
```

### Handler Changes Required

**Current Problem:** Handlers cache nodes dict in `__init__`:

```python
# agentflow/graph/utils/invoke_handler.py (CURRENT)
class InvokeHandler:
    def __init__(self, nodes: dict[str, Node], edges: list[Edge], ...):
        self.nodes = nodes  # ❌ Cached reference
    
    async def _execute_graph(self, state, config):
        node = self.nodes[current_node]  # ❌ Uses cached dict
```

**Solution:** Read from InjectQ on each node execution:

```python
# agentflow/graph/utils/invoke_handler.py (UPDATED)
class InvokeHandler:
    @inject
    def __init__(
        self,
        nodes: dict[str, Node],  # Still receive for validation
        edges: list[Edge],
        get_node_factory: Callable[[str], Node] = Inject["get_node"],  # NEW
        ...
    ):
        self.nodes = nodes  # Keep for validation/iteration
        self.edges = edges
        self._get_node = get_node_factory  # Store factory
    
    async def _execute_graph(self, state, config):
        # ✅ Read from InjectQ factory - picks up overrides
        node = self._get_node(current_node)
        result = await node.execute(config, state)
```

**Same change needed in `stream_handler.py`:**

```python
# agentflow/graph/utils/stream_handler.py (UPDATED)
class StreamHandler:
    @inject
    def __init__(
        self,
        nodes: dict[str, Node],
        edges: list[Edge],
        get_node_factory: Callable[[str], Node] = Inject["get_node"],
        ...
    ):
        self.nodes = nodes
        self.edges = edges
        self._get_node = get_node_factory  # Store factory
    
    async def _execute_graph(self, state, input_data, config):
        # ✅ Read from InjectQ
        node = self._get_node(current_node)
        result = await node.execute(config, state)
```

**Why This Works:**

1. **Compile time:** `state_graph.py` registers factory:
   ```python
   self._container.bind_factory("get_node", lambda x: self.nodes[x])
   ```

2. **Override after compile:** `CompiledGraph.override_node()` updates the dict AND re-binds the factory

3. **Handler execution:** Handlers call `self._get_node(name)` which invokes the factory, getting the latest node

**Key Insight:** The factory closure captures `self.nodes`, so when we update that dict, the factory returns the new node!

### Complete Testing Patterns

#### Pattern A: Graph Factory (Override After Compile)

```python
# production_workflows.py
def create_customer_service_workflow() -> CompiledGraph:
    """Production workflow factory."""
    graph = StateGraph()
    graph.add_node("CLASSIFY", classification_agent)
    graph.add_node("HANDLE", handler_agent)
    graph.add_node("TOOLS", production_tool_node)
    # ... complex routing logic
    return graph.compile(
        checkpointer=PgCheckpointer(),
        store=QdrantStore(),
    )

# test_customer_service.py
@pytest.mark.asyncio
async def test_customer_service_workflow():
    # Get pre-built production graph
    compiled = create_customer_service_workflow()
    
    # Override with test components AFTER compile
    compiled.override_node("CLASSIFY", test_classify_agent)
    compiled.override_node("HANDLE", test_handler_agent)
    compiled.override_node("TOOLS", test_tool_node)
    
    result = await compiled.ainvoke({"messages": [...]})
    assert "expected" in result["messages"][-1].text()
```

#### Pattern B: Build in Test (Override Before Compile)

```python
@pytest.mark.asyncio
async def test_simple_workflow():
    # Build graph in test
    graph = StateGraph()
    graph.add_node("MAIN", mock_agent)  # Already using mock
    graph.set_entry_point("MAIN")
    graph.add_edge("MAIN", END)
    
    # OR override before compile if needed
    # graph.override_node("MAIN", different_mock)
    
    compiled = graph.compile()
    result = await compiled.ainvoke(...)
```

#### Pattern C: Fixture Factory

```python
# conftest.py
@pytest.fixture
def compiled_workflow():
    """Fixture that returns a compiled test workflow."""
    graph = StateGraph()
    graph.add_node("MAIN", TestAgent(responses=["Test"]))
    graph.set_entry_point("MAIN")
    graph.add_edge("MAIN", END)
    return graph.compile()

@pytest.mark.asyncio
async def test_with_fixture(compiled_workflow):
    # Can still override after compile if needed
    compiled_workflow.override_node("MAIN", TestAgent(responses=["Different"]))
    
    result = await compiled_workflow.ainvoke(...)
```

### Why Both Patterns Matter

| Scenario | Pattern | Example |
|----------|---------|---------|
| Unit test for single node | Build in test | Test just one agent's behavior |
| Integration test | Override before compile | Test multiple components together |
| Testing production factory | Override after compile | Test actual production graph structure |
| Pytest fixture | Either | Depends on fixture design |

**Key Insight:** By supporting `override_node()` on BOTH `StateGraph` and `CompiledGraph`, users can choose the pattern that fits their testing needs.

---

## Part 8: Testing Helper Classes

### Recommended Testing Utilities

Based on common testing patterns, here are useful helper classes to include in `agentflow/testing/`:

#### 1. `TestAgent` (Already Covered)
- Mock agent with predefined responses
- Call tracking and assertions

#### 2. `TestContext` (Already Covered)
- Container isolation
- Graph factory
- Store management

#### 3. `MockToolRegistry` - Tool Mock Manager

```python
# agentflow/testing/mock_tools.py
class MockToolRegistry:
    """Registry for managing mock tools in tests.
    
    Simplifies creating and tracking mock tool calls.
    
    Example:
        ```python
        tools = MockToolRegistry()
        tools.register("get_weather", lambda city: f"Sunny in {city}")
        tools.register("send_email", lambda **kw: "Email sent")
        
        tool_node = ToolNode(list(tools.functions.values()))
        
        # After test
        assert tools.was_called("get_weather")
        assert tools.call_count("send_email") == 2
        ```
    """
    
    def __init__(self):
        self.functions: dict[str, callable] = {}
        self.calls: dict[str, list[dict]] = {}
    
    def register(self, name: str, mock_func: callable):
        """Register a mock tool function."""
        
        # Wrap to track calls
        def tracked_func(*args, **kwargs):
            if name not in self.calls:
                self.calls[name] = []
            self.calls[name].append({"args": args, "kwargs": kwargs})
            return mock_func(*args, **kwargs)
        
        tracked_func.__name__ = name
        self.functions[name] = tracked_func
    
    def was_called(self, name: str) -> bool:
        """Check if a tool was called."""
        return name in self.calls and len(self.calls[name]) > 0
    
    def call_count(self, name: str) -> int:
        """Get number of times a tool was called."""
        return len(self.calls.get(name, []))
    
    def get_calls(self, name: str) -> list[dict]:
        """Get all calls made to a tool."""
        return self.calls.get(name, [])
    
    def reset(self):
        """Clear all call history."""
        self.calls.clear()
```

#### 4. `GraphBuilder` - Fluent Test Graph Builder

```python
# agentflow/testing/graph_builder.py
class GraphBuilder:
    """Fluent builder for creating test graphs.
    
    Simplifies common test graph patterns.
    
    Example:
        ```python
        graph = (GraphBuilder()
            .with_agent("MAIN", TestAgent(responses=["Hi"]))
            .with_tools("TOOLS", [mock_weather])
            .route_to_tools("MAIN", "TOOLS")
            .build())
        ```
    """
    
    def __init__(self, state=None, container=None):
        from agentflow import StateGraph
        self.graph = StateGraph(state=state, container=container)
        self._entry_set = False
    
    def with_agent(self, name: str, agent: BaseAgent):
        """Add an agent node."""
        self.graph.add_node(name, agent)
        if not self._entry_set:
            self.graph.set_entry_point(name)
            self._entry_set = True
        return self
    
    def with_node(self, name: str, func: callable):
        """Add a function node."""
        self.graph.add_node(name, func)
        if not self._entry_set:
            self.graph.set_entry_point(name)
            self._entry_set = True
        return self
    
    def with_tools(self, name: str, tools: list):
        """Add a ToolNode."""
        from agentflow.graph.tool_node import ToolNode
        self.graph.add_node(name, ToolNode(tools))
        return self
    
    def connect(self, from_node: str, to_node: str):
        """Add edge between nodes."""
        self.graph.add_edge(from_node, to_node)
        return self
    
    def route_to_tools(self, agent_node: str, tool_node: str):
        """Add common agent→tools→agent pattern."""
        from agentflow.utils import END
        
        def router(state):
            last = state.context[-1] if state.context else None
            if last and hasattr(last, 'has_tool_calls') and last.has_tool_calls():
                return tool_node
            return END
        
        self.graph.add_conditional_edges(
            agent_node,
            router,
            {tool_node: tool_node, END: END}
        )
        self.graph.add_edge(tool_node, agent_node)
        return self
    
    def build(self) -> StateGraph:
        """Return the built graph."""
        return self.graph
    
    def compile(self, **kwargs) -> CompiledGraph:
        """Build and compile the graph."""
        return self.graph.compile(**kwargs)
```

#### 5. `MessageBuilder` - Fluent Message Creation

```python
# agentflow/testing/message_builder.py
class MessageBuilder:
    """Fluent builder for creating test messages.
    
    Example:
        ```python
        msg = (MessageBuilder()
            .user("What's the weather?")
            .assistant("Let me check")
            .tool_call("get_weather", city="NYC")
            .tool_result("call_123", "Sunny, 72°F")
            .build())
        ```
    """
    
    def __init__(self):
        from agentflow.state import Message
        self.messages: list[Message] = []
        self.Message = Message
    
    def user(self, content: str):
        """Add user message."""
        self.messages.append(
            self.Message.text_message(content, role="user")
        )
        return self
    
    def assistant(self, content: str):
        """Add assistant message."""
        self.messages.append(
            self.Message.text_message(content, role="assistant")
        )
        return self
    
    def tool_call(self, name: str, call_id: str = None, **args):
        """Add assistant message with tool call."""
        from agentflow.state import ToolCallBlock
        
        call_id = call_id or f"call_{len(self.messages)}"
        msg = self.Message(
            role="assistant",
            content=[ToolCallBlock(id=call_id, name=name, args=args)]
        )
        self.messages.append(msg)
        return self
    
    def tool_result(self, call_id: str, output: str):
        """Add tool result message."""
        self.messages.append(
            self.Message.tool_message(output, call_id=call_id)
        )
        return self
    
    def build(self) -> list:
        """Return list of messages."""
        return self.messages
    
    def as_input(self) -> dict:
        """Return as graph input dict."""
        return {"messages": self.messages}
```

#### 6. `AssertionHelper` - Common Test Assertions

```python
# agentflow/testing/assertions.py
class AssertionHelper:
    """Helper for common graph execution assertions.
    
    Example:
        ```python
        result = await compiled.ainvoke(...)
        
        assertions = AssertionHelper(result)
        assertions.has_messages(3)
        assertions.last_message_contains("weather")
        assertions.no_errors()
        ```
    """
    
    def __init__(self, result: dict):
        self.result = result
        self.messages = result.get("messages", [])
    
    def has_messages(self, count: int):
        """Assert exact message count."""
        actual = len(self.messages)
        assert actual == count, f"Expected {count} messages, got {actual}"
        return self
    
    def last_message_contains(self, text: str):
        """Assert last message contains text."""
        if not self.messages:
            raise AssertionError("No messages in result")
        
        last = self.messages[-1]
        content = last.text() if hasattr(last, 'text') else str(last.content)
        assert text.lower() in content.lower(), \
            f"Expected '{text}' in last message: {content}"
        return self
    
    def last_message_from(self, role: str):
        """Assert last message role."""
        if not self.messages:
            raise AssertionError("No messages in result")
        
        actual = self.messages[-1].role
        assert actual == role, f"Expected role '{role}', got '{actual}'"
        return self
    
    def no_errors(self):
        """Assert no error messages."""
        from agentflow.state import ErrorBlock
        
        for msg in self.messages:
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ErrorBlock):
                        raise AssertionError(f"Found error: {block.message}")
        return self
    
    def has_tool_calls(self):
        """Assert at least one message has tool calls."""
        for msg in self.messages:
            if hasattr(msg, 'has_tool_calls') and msg.has_tool_calls():
                return self
        raise AssertionError("No tool calls found in messages")
    
    def state_completed(self):
        """Assert execution completed successfully."""
        state = self.result.get("state")
        if state:
            status = state.execution_meta.status if hasattr(state, 'execution_meta') else None
            assert status == ExecutionStatus.COMPLETED, \
                f"Expected COMPLETED status, got {status}"
        return self
```

### Summary of Testing Helpers

| Helper | Purpose | Complexity |
|--------|---------|------------|
| `TestAgent` | Mock LLM responses | Simple |
| `TestContext` | Test environment isolation | Simple |
| `MockToolRegistry` | Track tool calls | Medium |
| `GraphBuilder` | Fluent graph creation | Medium |
| `MessageBuilder` | Fluent message creation | Simple |
| `AssertionHelper` | Common assertions | Simple |

**Recommended Minimal Set:**
- ✅ `TestAgent`
- ✅ `TestContext`
- ✅ `MockToolRegistry`

**Optional But Useful:**
- `GraphBuilder` - For complex graph tests
- `MessageBuilder` - For conversation tests
- `AssertionHelper` - For cleaner assertions

---

## Summary: What Gets Implemented

### New Files

| File | Purpose |
|------|---------|
| `agentflow/graph/base_agent.py` | Abstract BaseAgent class |
| `agentflow/testing/__init__.py` | TestContext, exports |
| `agentflow/testing/test_agent.py` | TestAgent implementation |
| `agentflow/testing/mock_tools.py` | MockToolRegistry for tool call tracking |
| `agentflow/testing/graph_builder.py` | GraphBuilder fluent API (optional) |
| `agentflow/testing/message_builder.py` | MessageBuilder fluent API (optional) |
| `agentflow/testing/assertions.py` | AssertionHelper for common checks (optional) |
| `agentflow/store/in_memory_store.py` | InMemoryStore for testing |

### Modified Files

| File | Changes |
|------|---------|
| `agentflow/graph/agent.py` | Inherit from BaseAgent, extract `_call_llm()` |
| `agentflow/graph/state_graph.py` | Add `override_node()` method |
| `agentflow/graph/compiled_graph.py` | Add `override_node()` method for post-compile overrides |
| `agentflow/graph/utils/invoke_handler.py` | Inject `get_node` factory, read nodes from InjectQ |
| `agentflow/graph/utils/stream_handler.py` | Inject `get_node` factory, read nodes from InjectQ |
| `agentflow/graph/__init__.py` | Export BaseAgent |
| `agentflow/__init__.py` | Export testing module |

### Public API Changes

```python
# New exports
from agentflow.graph import BaseAgent
from agentflow.testing import (
    TestAgent,
    TestContext,
    MockToolRegistry,
    GraphBuilder,      # Optional
    MessageBuilder,    # Optional
    AssertionHelper,   # Optional
)
from agentflow.store import InMemoryStore

# New methods
graph.override_node("NODE_NAME", test_function)      # Before compile
compiled.override_node("NODE_NAME", test_function)   # After compile
```

### Critical Implementation Detail: Handler Node Resolution

**Before:**
```python
# Handlers cached nodes dict
class InvokeHandler:
    def __init__(self, nodes: dict, ...):
        self.nodes = nodes  # ❌ Static reference
    
    async def _execute_graph(self, ...):
        node = self.nodes[current_node]  # ❌ Won't see overrides
```

**After:**
```python
# Handlers read from InjectQ factory
class InvokeHandler:
    @inject
    def __init__(
        self,
        nodes: dict,
        get_node_factory: Callable = Inject["get_node"],  # ✅ DI factory
        ...
    ):
        self.nodes = nodes
        self._get_node = get_node_factory  # ✅ Store factory
    
    async def _execute_graph(self, ...):
        node = self._get_node(current_node)  # ✅ Sees overrides!
```

This ensures `compiled.override_node()` works correctly!

### Testing Patterns Summary

| Pattern | When to Use | Override Timing |
|---------|-------------|-----------------|
| Build graph in test | Simple unit tests | Before or not needed |
| Override before compile | Control graph creation | Before compile |
| Override after compile | Pre-built production factories | After compile |
| Mock tool sources | ToolNode testing | N/A - pass to ToolNode |

---

## Why This Approach Is Better

| Previous Approach | New Approach |
|-------------------|--------------|
| Force users to inject `llm_completion` | Optional - use TestAgent or override_node |
| Complex MockLLMCompletion class | Simple TestAgent with responses list |
| Required changing production code | Production code unchanged |
| LLMService protocol abstraction | Direct inheritance, no protocols |
| NodeFactory pattern | Simple dict override |

**Key Insight:** The simplest solution is often the best. Users can:
1. Use `TestAgent` as drop-in replacement for `Agent`
2. Use `override_node()` to swap any function
3. Write tests without learning complex patterns
