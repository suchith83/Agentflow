# Comprehensive Analysis: Node-Level `Agent` Wrapper Class

## Executive Summary

This document provides an in-depth analysis of creating a **node-level** `Agent` wrapper class for the 10xScale Agentflow framework, rather than wrapping the entire graph. After extensive research into similar frameworks (LangGraph, CrewAI) and thorough codebase analysis, this approach offers the best balance of flexibility and convenience.

**Key Decision:** Create an `Agent` class that acts as a **smart node function** that can be used within existing graph structures, not a replacement for the graph itself.

---

## Problem Analysis

### Current Framework Status

The 10xScale Agentflow framework is powerful and flexible with a "framework agnostic" philosophy, but this comes at the cost of **significant boilerplate** for common use cases.

### Identified Pain Points

After analyzing `react_sync.py`, `react_di.py`, `react_di2.py`, and `react-mcp.py`, the following repetitive patterns emerge:

#### 1. **Repetitive `main_agent` Logic** (Every Example)
Every agent implementation requires the same 5-step pattern:
```python
async def main_agent(state: AgentState, config: dict):
    # Step 1: Define system prompt
    prompts = "You are a helpful assistant..."
    
    # Step 2: Convert state to messages
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )
    
    # Step 3: Get tools (if applicable)
    tools = await tool_node.all_tools()
    
    # Step 4: Call LLM
    response = await acompletion(
        model="gemini/gemini-2.5-flash",
        messages=messages,
        tools=tools,
    )
    
    # Step 5: Convert response back
    return ModelResponseConverter(response, converter="litellm")
```

This appears **identically** in all 4+ example files analyzed.

#### 2. **Manual Tool Routing Logic** (Every Example)
The `should_use_tools` function is **copy-pasted** across examples:
```python
def should_use_tools(state: AgentState) -> str:
    if not state.context or len(state.context) == 0:
        return "TOOL"
    last_message = state.context[-1]
    if (hasattr(last_message, "tools_calls") and 
        last_message.tools_calls and 
        len(last_message.tools_calls) > 0 and 
        last_message.role == "assistant"):
        return "TOOL"
    if last_message.role == "tool":
        return "MAIN"
    return END
```

#### 3. **Complex Graph Setup** (Every Example)
Setting up even a simple ReAct agent requires:
```python
graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)
graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")
app = graph.compile(checkpointer=checkpointer)
```

#### 4. **Model Converter Verbosity**
Users must understand and use `ModelResponseConverter` with the correct converter string:
```python
return ModelResponseConverter(response, converter="litellm")
```

#### 5. **No Built-in Learning/Memory**
Advanced features like RAG-based learning require manual:
- Store integration
- Query retrieval logic
- Context injection
- Memory saving logic

#### 6. **Streaming Complexity**
Streaming requires additional conditional logic:
```python
if state.context and len(state.context) > 0 and state.context[-1].role == "tool":
    # No tools on final response
    response = await acompletion(model=..., messages=messages)
else:
    # Tools available
    response = await acompletion(model=..., messages=messages, tools=tools)
```

---

## Research: How Other Frameworks Handle This

### 1. **LangGraph** (Closest Competitor)
- **Approach:** Low-level, does NOT abstract nodes
- **Philosophy:** "LangGraph does not abstract prompts or architecture"
- **Node Pattern:** Users write functions like we do:
  ```python
  def mock_llm(state: MessagesState):
      return {"messages": [{"role": "ai", "content": "hello world"}]}
  ```
- **Why:** They prefer transparency and flexibility over convenience
- **Learning:** They also have boilerplate but accept it as a trade-off

### 2. **CrewAI** (High-Level Approach)
- **Approach:** High-level `Agent` class with extensive configuration
- **Philosophy:** Opinionated, batteries-included
- **Agent Pattern:**
  ```python
  agent = Agent(
      role="Research Analyst",
      goal="Find and summarize information",
      backstory="Experienced researcher",
      tools=[SerperDevTool()],
      llm="gpt-4",
      memory=True,
      verbose=True,
      # ... 30+ parameters
  )
  ```
- **Why:** They target non-technical users and prioritize ease of use
- **Trade-off:** Less flexibility, harder to customize beyond preset options

### 3. **Our Position**
We sit between LangGraph (too low-level) and CrewAI (too opinionated). Our goal:
- **More convenient than LangGraph** (reduce boilerplate)
- **More flexible than CrewAI** (maintain graph control)

---

## Proposed Solution: Node-Level `Agent` Wrapper

### Core Concept

Instead of wrapping the **entire graph** (which would hide the powerful graph abstraction), we create an `Agent` class that acts as a **smart node function** that can be dropped into existing graphs.

### Why Node-Level, Not Graph-Level?

| Aspect | Graph-Level Wrapper | Node-Level Wrapper |
|--------|---------------------|-------------------|
| **Flexibility** | ❌ Hides graph structure | ✅ Works with existing graphs |
| **Learning Curve** | ⚠️ New mental model | ✅ Familiar to current users |
| **Multi-Agent** | ❌ Hard to compose | ✅ Easy to compose multiple agents |
| **Custom Flows** | ❌ Requires "ejecting" | ✅ Mix Agent nodes with custom nodes |
| **Debugging** | ❌ Opaque execution | ✅ Graph visualization still works |
| **Migration** | ⚠️ Breaking change | ✅ Gradual adoption |

### Key Insight from Codebase Analysis

Looking at `agentflow/prebuilt/agent/react.py`, we already have a pattern for this:
```python
class ReactAgent:
    def compile(self, main_node: Callable, tool_node: Callable):
        self._graph.add_node("MAIN", main_node)
        self._graph.add_node("TOOL", tool_node)
        # ... setup edges
```

The `ReactAgent` expects a **node function**, not a wrapper. We should follow this pattern but make the node function itself smarter.

---

## Detailed Design

### 1. The `Agent` Class (Node-Level Wrapper)

```python
class Agent:
    """
    A smart node function wrapper that handles LLM interactions with built-in
    message conversion, tool handling, streaming, and optional learning.
    
    This class is designed to be used as a node within a StateGraph, not as
    a replacement for the graph itself.
    
    Example:
        # Create an agent node
        agent = Agent(
            model="gemini/gemini-2.0-flash",
            system_prompt="You are a helpful assistant",
            tools=[weather_tool],
            learning=True
        )
        
        # Use it in a graph
        graph = StateGraph()
        graph.add_node("MAIN", agent)  # <-- Agent acts as a node function
        graph.add_node("TOOL", tool_node)
        graph.add_edge("MAIN", "TOOL")
        # ...
    """
    
    def __init__(
        self,
        model: str,
        system_prompt: str | list[dict[str, Any]] = "",
        tools: list[Callable] | ToolNode | None = None,
        learning: bool = False,
        store: BaseStore | None = None,
        streaming: bool = True,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        # LiteLLM-specific params
        **llm_kwargs,
    ):
        """
        Initialize an Agent node.
        
        Args:
            model: LiteLLM model string (e.g., "gpt-4", "gemini/gemini-2.0-flash")
            system_prompt: System prompt string or list of system message dicts
            tools: List of tool functions, ToolNode instance, or None
            learning: Enable automatic RAG-based learning
            store: BaseStore instance for learning (required if learning=True)
            streaming: Whether to support streaming responses
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens to generate
            **llm_kwargs: Additional LiteLLM parameters (top_p, etc.)
        """
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.learning = learning
        self.store = store
        self.streaming = streaming
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_kwargs = llm_kwargs
        
        # Internal setup
        self._tool_node = self._setup_tools()
        self._validate_config()
    
    def _setup_tools(self) -> ToolNode | None:
        """Convert tools to ToolNode if needed."""
        if self.tools is None:
            return None
        if isinstance(self.tools, ToolNode):
            return self.tools
        return ToolNode(self.tools)
    
    def _validate_config(self):
        """Validate configuration."""
        if self.learning and self.store is None:
            raise ValueError("store must be provided when learning=True")
    
    async def __call__(
        self,
        state: AgentState,
        config: dict[str, Any],
        # Injectable dependencies (optional)
        store: BaseStore | None = Inject[BaseStore],
        context_manager: BaseContextManager | None = Inject[BaseContextManager],
        **kwargs,
    ) -> ModelResponseConverter:
        """
        Execute the agent node.
        
        This method is called by the graph when this node is executed.
        It handles all the boilerplate: message conversion, LLM calls,
        tool handling, learning, etc.
        
        Args:
            state: Current agent state
            config: Execution configuration
            store: Injected store (if available)
            context_manager: Injected context manager (if available)
            **kwargs: Additional injected dependencies
        
        Returns:
            ModelResponseConverter wrapping the LLM response
        """
        # Step 1: Handle learning (retrieval)
        injected_context = ""
        if self.learning:
            store_instance = self.store or store
            if store_instance:
                injected_context = await self._retrieve_knowledge(
                    state, config, store_instance
                )
        
        # Step 2: Build system prompt with injected context
        system_messages = self._build_system_prompt(injected_context)
        
        # Step 3: Convert state to messages
        messages = convert_messages(
            system_prompts=system_messages,
            state=state,
        )
        
        # Step 4: Get tools (if applicable)
        tools_schema = None
        if self._tool_node:
            # Check if last message is a tool result
            # If so, don't include tools (final response)
            if not (state.context and 
                    len(state.context) > 0 and 
                    state.context[-1].role == "tool"):
                tools_schema = await self._tool_node.all_tools()
        
        # Step 5: Call LLM
        llm_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **self.llm_kwargs,
        }
        if self.max_tokens:
            llm_params["max_tokens"] = self.max_tokens
        if tools_schema:
            llm_params["tools"] = tools_schema
        
        # Import here to avoid circular dependency
        from litellm import acompletion
        response = await acompletion(**llm_params)
        
        # Step 6: Handle learning (storage) - fire and forget
        if self.learning and store_instance:
            asyncio.create_task(
                self._store_interaction(state, response, config, store_instance)
            )
        
        # Step 7: Return wrapped response
        return ModelResponseConverter(response, converter="litellm")
    
    async def _retrieve_knowledge(
        self,
        state: AgentState,
        config: dict[str, Any],
        store: BaseStore,
    ) -> str:
        """Retrieve relevant knowledge from store."""
        if not state.context or len(state.context) == 0:
            return ""
        
        # Get last user message as query
        last_user_msg = None
        for msg in reversed(state.context):
            if msg.role == "user":
                last_user_msg = msg
                break
        
        if not last_user_msg:
            return ""
        
        query = last_user_msg.text()
        
        # Search store
        try:
            results = await store.asearch(
                config=config,
                query=query,
                limit=3,
            )
            
            if not results:
                return ""
            
            # Format results
            context_parts = []
            for idx, result in enumerate(results, 1):
                context_parts.append(
                    f"[Memory {idx}] {result.content[:200]}..."
                )
            
            return "\n\n".join(context_parts)
        except Exception as e:
            logger.warning(f"Failed to retrieve knowledge: {e}")
            return ""
    
    async def _store_interaction(
        self,
        state: AgentState,
        response: Any,
        config: dict[str, Any],
        store: BaseStore,
    ):
        """Store the interaction in the knowledge base (fire and forget)."""
        try:
            # Get last user message
            last_user_msg = None
            for msg in reversed(state.context):
                if msg.role == "user":
                    last_user_msg = msg
                    break
            
            if not last_user_msg:
                return
            
            # Extract assistant response
            assistant_content = ""
            if hasattr(response, "choices") and response.choices:
                assistant_content = response.choices[0].message.content or ""
            
            if not assistant_content:
                return
            
            # Store as Q&A pair
            content = f"Q: {last_user_msg.text()}\nA: {assistant_content}"
            
            await store.astore(
                config=config,
                content=content,
                memory_type=MemoryType.EPISODIC,
                category="agent_interactions",
                metadata={
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to store interaction: {e}")
    
    def _build_system_prompt(self, injected_context: str = "") -> list[dict[str, Any]]:
        """Build system prompt messages with optional injected context."""
        if isinstance(self.system_prompt, list):
            messages = self.system_prompt.copy()
        else:
            messages = [{"role": "system", "content": self.system_prompt}]
        
        # Inject context if available
        if injected_context:
            context_msg = {
                "role": "system",
                "content": f"\n\n## Relevant Context from Memory:\n{injected_context}",
            }
            messages.append(context_msg)
        
        return messages
    
    def get_tool_node(self) -> ToolNode | None:
        """Get the associated ToolNode for use in graph setup."""
        return self._tool_node
```

### 2. Usage Patterns

#### Pattern 1: Simple ReAct Agent (Replaces Current Boilerplate)

**Before (56 lines):**
```python
def get_weather(location: str): ...

tool_node = ToolNode([get_weather])

async def main_agent(state: AgentState):
    prompts = "You are a helpful assistant..."
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": prompts}],
        state=state,
    )
    tools = await tool_node.all_tools()
    if state.context and len(state.context) > 0 and state.context[-1].role == "tool":
        response = await acompletion(model="gemini/gemini-2.5-flash", messages=messages)
    else:
        response = await acompletion(model="gemini/gemini-2.5-flash", messages=messages, tools=tools)
    return ModelResponseConverter(response, converter="litellm")

def should_use_tools(state: AgentState) -> str:
    if not state.context or len(state.context) == 0:
        return "TOOL"
    last_message = state.context[-1]
    if (hasattr(last_message, "tools_calls") and ...):
        return "TOOL"
    if last_message.role == "tool":
        return "MAIN"
    return END

graph = StateGraph()
graph.add_node("MAIN", main_agent)
graph.add_node("TOOL", tool_node)
graph.add_conditional_edges("MAIN", should_use_tools, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")
app = graph.compile(checkpointer=checkpointer)
```

**After (15 lines):**
```python
def get_weather(location: str): ...

agent = Agent(
    model="gemini/gemini-2.5-flash",
    system_prompt="You are a helpful assistant...",
    tools=[get_weather],
)

# Use with existing ReactAgent pattern
react = ReactAgent()
app = react.compile(
    main_node=agent,
    tool_node=agent.get_tool_node(),
    checkpointer=checkpointer,
)
```

#### Pattern 2: Agent with Learning

```python
from agentflow.store import QdrantStore

store = QdrantStore(...)
await store.asetup()

agent = Agent(
    model="gpt-4",
    system_prompt="You are a personalized assistant",
    tools=[search_tool, calculator],
    learning=True,  # 🔑 Enable automatic learning
    store=store,
)

# The agent will:
# 1. Retrieve relevant past interactions before responding
# 2. Automatically store new Q&A pairs for future reference
react = ReactAgent()
app = react.compile(main_node=agent, tool_node=agent.get_tool_node())
```

#### Pattern 3: Multi-Agent System (Mix Agent nodes with custom nodes)

```python
# Create specialized agents
researcher = Agent(
    model="gpt-4",
    system_prompt="You are a research expert",
    tools=[search_tool],
)

writer = Agent(
    model="claude-3-sonnet",
    system_prompt="You are a creative writer",
)

# Custom node for validation
def validate_output(state: AgentState, config: dict):
    # Custom logic
    return state

# Mix agents with custom nodes in graph
graph = StateGraph()
graph.add_node("RESEARCH", researcher)
graph.add_node("RESEARCH_TOOLS", researcher.get_tool_node())
graph.add_node("VALIDATE", validate_output)
graph.add_node("WRITE", writer)

# Custom flow
graph.add_edge(START, "RESEARCH")
graph.add_conditional_edges("RESEARCH", should_use_tools, 
                           {"TOOLS": "RESEARCH_TOOLS", "NEXT": "VALIDATE"})
graph.add_edge("RESEARCH_TOOLS", "RESEARCH")
graph.add_edge("VALIDATE", "WRITE")
graph.add_edge("WRITE", END)

app = graph.compile()
```

### 3. What We Gain

| Feature | Before | After |
|---------|--------|-------|
| **Lines of Code** | ~60 lines per agent | ~10 lines per agent |
| **Message Conversion** | Manual `convert_messages()` | ✅ Automatic |
| **LLM Call** | Manual `acompletion()` | ✅ Automatic |
| **Model Converter** | Manual `ModelResponseConverter` | ✅ Automatic |
| **Tool Handling** | Manual conditional logic | ✅ Automatic |
| **Learning/RAG** | 50+ lines to implement | ✅ `learning=True` |
| **Streaming** | Complex conditional logic | ✅ Built-in support |
| **Graph Visibility** | ✅ Full control | ✅ Full control (same) |
| **Multi-Agent** | ⚠️ Copy-paste code | ✅ Reusable agents |

### 4. What We Lose (Trade-offs)

#### ❌ Framework Agnosticism

**Current State:**
- Users can use any LLM library (OpenAI SDK, Anthropic SDK, LangChain, etc.)
- Adapters provide conversion from different model responses

**After Agent Class:**
- **Locked into LiteLLM** - The `Agent` class uses `litellm.acompletion` internally
- Users who want to use other libraries must write custom node functions (can't use `Agent`)

**Mitigation:**
- LiteLLM itself supports 100+ models (OpenAI, Anthropic, Google, Azure, AWS, etc.)
- Advanced users can still write custom node functions for non-LiteLLM use cases
- We can provide an `adapter` parameter later if needed

#### ⚠️ Less Fine-Grained Control

**Current State:**
- Full control over message formatting
- Can customize every LLM parameter
- Can insert custom logic between steps

**After Agent Class:**
- System prompt customization only
- Limited LLM parameters exposed (temperature, max_tokens, **kwargs)
- No control over message conversion logic

**Mitigation:**
- Expose common parameters (temperature, top_p, max_tokens, etc.) via `**llm_kwargs`
- Advanced users can still use custom node functions
- Can add hooks for customization later (e.g., `message_preprocessor` callback)

#### ⚠️ Opinionated Learning Strategy

**Current State:**
- Complete freedom in how to implement learning
- Can use any retrieval/storage strategy

**After Agent Class:**
- Fixed learning pattern: retrieve before, store after
- Stores Q&A pairs automatically
- No control over what gets stored or when

**Mitigation:**
- `learning=False` by default (opt-in)
- Can still use custom nodes for advanced learning patterns
- Document the learning strategy clearly

### 5. Framework Agnosticism: Do We Need It?

#### Arguments FOR Keeping It

1. **Flexibility:** Users with existing code using specific SDKs can integrate easily
2. **Bleeding Edge:** Some models release features only in their native SDKs first
3. **Control:** Power users want fine-grained control over every parameter
4. **Marketing:** "Framework agnostic" is a selling point

#### Arguments AGAINST Keeping It

1. **Maintenance Burden:** Supporting multiple adapters is complex
2. **User Confusion:** Too many choices paralyze new users
3. **LiteLLM Coverage:** LiteLLM supports 100+ models already (good enough for 95% of use cases)
4. **Boilerplate:** Framework agnosticism is causing the repetitive code problem
5. **Industry Trend:** Most successful frameworks are opinionated (LangChain uses their abstractions, CrewAI has their own, etc.)

#### Recommendation: **Hybrid Approach**

```
┌─────────────────────────────────────────────────┐
│         10xScale Agentflow Framework            │
├─────────────────────────────────────────────────┤
│  High-Level (Opinionated)                       │
│  ┌───────────────────────────────────────────┐  │
│  │ Agent Class (LiteLLM-based)               │  │
│  │ - 95% of use cases                        │  │
│  │ - Built-in learning, streaming            │  │
│  │ - Minimal boilerplate                     │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  Low-Level (Flexible)                           │
│  ┌───────────────────────────────────────────┐  │
│  │ Custom Node Functions                     │  │
│  │ - Power users & edge cases                │  │
│  │ - Full control                            │  │
│  │ - Use any LLM library                     │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  ┌───────────────────────────────────────────┐  │
│  │ StateGraph (Foundation)                   │  │
│  │ - Always available                        │  │
│  │ - Framework agnostic                      │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

**Strategy:**
1. **Keep the foundation framework-agnostic** (StateGraph, Node, etc.)
2. **Add opinionated convenience layers** (Agent class)
3. **Document both paths clearly:**
   - Quick Start: Use `Agent` class
   - Advanced: Use custom node functions

This gives us:
- ✅ Convenience for common cases (Agent class)
- ✅ Flexibility for power users (custom nodes)
- ✅ Clean architecture (layers of abstraction)

---

## Implementation Plan

### Phase 1: Core Agent Class (Week 1)

#### 1.1 Create `agentflow/agent.py`
```python
class Agent:
    """Smart node function wrapper for LLM interactions."""
    
    def __init__(self, model, system_prompt, tools, learning, ...): ...
    async def __call__(self, state, config, **deps): ...
    def get_tool_node(self): ...
```

#### 1.2 Implement Core Features
- ✅ System prompt handling
- ✅ Message conversion (use existing `convert_messages`)
- ✅ LiteLLM integration
- ✅ Tool handling (conditional logic for final response)
- ✅ ModelResponseConverter integration
- ✅ Dependency injection support

#### 1.3 Write Tests
```python
# tests/agent/test_agent_basic.py
async def test_agent_basic_call():
    agent = Agent(model="gpt-4", system_prompt="Test")
    state = AgentState(...)
    result = await agent(state, {})
    assert isinstance(result, ModelResponseConverter)

async def test_agent_with_tools():
    agent = Agent(model="gpt-4", tools=[my_tool])
    # Test tool calling flow
```

### Phase 2: Learning Integration (Week 2)

#### 2.1 Implement Learning Methods
```python
# In Agent class
async def _retrieve_knowledge(self, state, config, store): ...
async def _store_interaction(self, state, response, config, store): ...
```

#### 2.2 Store Integration
- Use existing `BaseStore` interface
- Test with `QdrantStore` and `Mem0Store`
- Handle errors gracefully (log warnings, don't crash)

#### 2.3 Write Learning Tests
```python
# tests/agent/test_agent_learning.py
async def test_agent_learning_retrieval():
    store = MockStore()
    agent = Agent(learning=True, store=store)
    # Test that retrieve is called

async def test_agent_learning_storage():
    store = MockStore()
    agent = Agent(learning=True, store=store)
    # Test that store is called after response
```

### Phase 3: Documentation & Examples (Week 3)

#### 3.1 Update Documentation
- Add "Agent Class" section to docs
- Document all parameters
- Show comparison before/after
- Document trade-offs clearly

#### 3.2 Create Examples
```
examples/agent/
├── simple_agent.py          # Basic usage
├── agent_with_tools.py      # Tool calling
├── agent_with_learning.py   # RAG/memory
├── multi_agent_system.py    # Multiple agents
└── custom_vs_agent.py       # When to use custom nodes
```

#### 3.3 Update Existing Examples
- Refactor `examples/react/react_sync.py` to show Agent usage
- Keep old version as `react_sync_manual.py` for comparison
- Add comments explaining the reduction in code

### Phase 4: Migration Guide (Week 4)

#### 4.1 Write Migration Guide
```markdown
# Migrating to Agent Class

## Should You Migrate?

✅ Migrate if:
- You're writing a simple ReAct agent
- You want built-in learning/memory
- You want less boilerplate

❌ Don't migrate if:
- You need a non-LiteLLM library
- You need custom message formatting
- You need fine-grained control between steps

## Migration Steps

### Before
[Show old code]

### After
[Show new code]

## Hybrid Approach
[Show mixing Agent nodes with custom nodes]
```

#### 4.2 Create Codemods (Optional)
- Script to help convert old code to new pattern
- Detect `main_agent` functions and suggest Agent replacement

---

## Pros and Cons Analysis

### ✅ Pros

#### 1. **Massive Boilerplate Reduction**
- 60 lines → 10 lines for typical agent
- Developers can focus on business logic, not plumbing

#### 2. **Built-in Best Practices**
- Correct tool handling logic
- Proper message conversion
- Optimal streaming setup

#### 3. **Learning Made Easy**
- `learning=True` gives automatic RAG
- No need to understand vector stores to get started
- Can still customize for advanced users

#### 4. **Better DX (Developer Experience)**
- Less cognitive load
- Faster prototyping
- Easier to onboard new users

#### 5. **Maintains Graph Flexibility**
- Agent is just a node, not a graph wrapper
- Can mix Agent nodes with custom nodes
- Full graph visualization still works

#### 6. **Natural Evolution**
- Follows existing `ReactAgent` pattern
- Doesn't break existing code
- Gradual adoption path

### ❌ Cons

#### 1. **Loss of Framework Agnosticism**
- Locked into LiteLLM for Agent class
- Users with other SDKs must use custom nodes
- **Mitigation:** LiteLLM supports 100+ models; custom nodes still available

#### 2. **Less Fine-Grained Control**
- Can't customize message conversion logic
- Limited LLM parameter exposure
- **Mitigation:** Expose common params; document custom node path

#### 3. **Opinionated Learning Strategy**
- Fixed retrieve/store pattern
- No control over what/when to store
- **Mitigation:** `learning=False` by default; document behavior clearly

#### 4. **Increased Maintenance**
- New class to maintain
- More tests to write
- More docs to update
- **Mitigation:** Well-defined scope; good test coverage

#### 5. **Potential Lock-in**
- Users might over-rely on Agent class
- Harder to switch to custom implementation later
- **Mitigation:** Document custom node patterns prominently

### ⚖️ Overall Assessment

| Aspect | Weight | Score (1-10) | Weighted |
|--------|--------|--------------|----------|
| **DX Improvement** | 30% | 10 | 3.0 |
| **Boilerplate Reduction** | 25% | 10 | 2.5 |
| **Flexibility Loss** | 20% | 6 | 1.2 |
| **Maintenance Cost** | 15% | 7 | 1.05 |
| **Learning Curve** | 10% | 9 | 0.9 |
| **TOTAL** | | | **8.65/10** |

**Conclusion:** The benefits significantly outweigh the costs. The Agent class will make the framework more accessible while maintaining the powerful graph abstraction for advanced users.

---

## Migration Strategy

### For New Users

**Recommendation:** Start with `Agent` class

```python
from agentflow import Agent, ReactAgent

agent = Agent(
    model="gpt-4",
    system_prompt="You are helpful",
    tools=[my_tool],
)

react = ReactAgent()
app = react.compile(main_node=agent, tool_node=agent.get_tool_node())
```

### For Existing Users

#### Option 1: Gradual Migration (Recommended)

1. **Keep existing agents as-is** (they still work)
2. **Use Agent class for new agents**
3. **Refactor old agents when touching that code**

```python
# Old agents (still work)
async def old_agent(state, config):
    # ... manual logic

# New agents (use Agent class)
new_agent = Agent(model="gpt-4", ...)

# Mix in same graph
graph.add_node("OLD", old_agent)
graph.add_node("NEW", new_agent)
```

#### Option 2: Full Refactor

1. Identify all `main_agent` functions
2. Replace with `Agent` instances
3. Test thoroughly
4. Deploy

**Estimation:** 
- Simple agent: 15 minutes
- Agent with learning: 30 minutes
- Multi-agent system: 1-2 hours

### Breaking Changes

**None.** The Agent class is purely additive.

- ✅ All existing code continues to work
- ✅ No API changes to StateGraph
- ✅ No changes to Node, ToolNode, etc.

### Deprecation Timeline

**No deprecations planned.**

- Custom node functions are still first-class citizens
- Documentation will cover both approaches
- Agent class is a convenience layer, not a replacement

---

## Comparison with Other Frameworks

### vs. LangGraph

| Feature | LangGraph | AgentFlow (Current) | AgentFlow (with Agent) |
|---------|-----------|---------------------|------------------------|
| **Boilerplate** | High | High | Low |
| **Graph Control** | Full | Full | Full |
| **Learning** | Manual | Manual | `learning=True` |
| **Multi-Agent** | Manual | Manual | Mix Agent nodes |
| **Abstraction Level** | Low | Low | Low + High layers |

**Our Advantage:** We offer **both** low-level control and high-level convenience.

### vs. CrewAI

| Feature | CrewAI | AgentFlow (with Agent) |
|---------|--------|------------------------|
| **Ease of Use** | Very High | High |
| **Flexibility** | Low | High |
| **Graph Visibility** | Hidden | Full |
| **Custom Nodes** | Hard | Easy |
| **Learning Curve** | Very Low | Low |

**Our Advantage:** More flexible while maintaining ease of use.

---

## Alternative Approaches Considered

### Alternative 1: Keep Current Design (No Wrapper)

**Pros:**
- No new code to write
- Full transparency
- Complete control

**Cons:**
- Boilerplate remains
- Learning still manual
- New users struggle

**Verdict:** ❌ Doesn't solve the problem

### Alternative 2: Graph-Level Wrapper

```python
agent = Agent(model="gpt-4", tools=[...])
result = agent.invoke("Hello")  # Hides the graph
```

**Pros:**
- Very simple API
- Similar to OpenAI Assistants API

**Cons:**
- Hides graph structure
- Hard to do multi-agent
- Debugging is opaque
- Can't visualize flow

**Verdict:** ❌ Too opinionated, loses key value prop

### Alternative 3: Multiple Wrapper Types

```python
# Different wrappers for different use cases
agent = SimpleAgent(...)      # For beginners
agent = ReactAgent(...)       # For tool use
agent = LearningAgent(...)    # For RAG
agent = StreamingAgent(...)   # For streaming
```

**Pros:**
- Targeted for specific use cases

**Cons:**
- Too many classes to learn
- Overlap and confusion
- Maintenance nightmare

**Verdict:** ❌ Too complex

### Alternative 4: Node-Level Wrapper (CHOSEN)
        self,
        model: str,
        tools: list[Callable] | None = None,
        system_prompt: str = "",
        learning: bool = False,
        store: BaseStore | None = None,
        checkpointer: BaseCheckpointer | None = None,
        # ... other config
    ):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.learning = learning
        self.store = store or (InMemoryStore() if learning else None)
        
        # Internal components
        self.tool_node = ToolNode(tools) if tools else None
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledGraph:
        graph = StateGraph()
        graph.add_node("MAIN", self._main_node)
        
        if self.tools:
            graph.add_node("TOOL", self.tool_node)
            graph.add_conditional_edges(
                "MAIN",
                self._should_use_tools,
                {"TOOL": "TOOL", END: END}
            )
            graph.add_edge("TOOL", "MAIN")
        else:
            graph.add_edge("MAIN", END)
            
        graph.set_entry_point("MAIN")
        return graph.compile(checkpointer=self.checkpointer)

    async def _main_node(self, state: AgentState, config: dict):
        # 1. Retrieve Knowledge (if learning=True)
        injected_context = ""
        if self.learning and self.store:
            # Query store based on last user message
            relevant_docs = await self.store.search(state.last_user_message)
            injected_context = self._format_docs(relevant_docs)

        # 2. Prepare Messages
        full_system_prompt = f"{self.system_prompt}\n\nContext:\n{injected_context}"
        messages = convert_messages(
            system_prompts=[{"role": "system", "content": full_system_prompt}],
            state=state
        )

        # 3. Call LLM
        tools_schema = await self.tool_node.all_tools() if self.tool_node else None
        response = await acompletion(
            model=self.model,
            messages=messages,
            tools=tools_schema
        )

        # 4. Save Knowledge (if learning=True)
        # This is tricky: When to save? 
        # Option A: Save every User-Assistant pair.
        # Option B: Use a separate "Reflector" call to extract facts.
        # For v1, we might just save the interaction history or rely on explicit "save" tools.
        if self.learning and self.store:
             await self.store.add(state.last_user_message, response.content)

        return ModelResponseConverter(response, converter="litellm")

    def _should_use_tools(self, state: AgentState) -> str:
        # Standard logic: if last message has tool_calls, return "TOOL", else END
        ...
```

### 2. Pros & Cons

**Pros:**
-   **Productivity:** Reduces setup time from minutes to seconds.
-   **Consistency:** Enforces best practices for tool handling and loops.
-   **Power:** Makes advanced features (RAG/Memory) accessible to beginners.
-   **Readability:** User code focuses on business logic (tools/prompts), not plumbing.

**Cons:**
-   **Coupling:** Tightly couples the framework to `litellm`. (However, `litellm` itself is model-agnostic, so this is a minor concern).
-   **Opacity:** Hides the graph structure. Users might find it harder to debug if they don't understand the underlying graph.
-   **Flexibility:** Harder to customize the graph (e.g., adding a "Human in the loop" node in the middle) without "ejecting" from the `Agent` class.

### 3. "Learning" Implementation Strategy

To implement `learning=True` effectively:
1.  **Retrieval:** We need a standard way to query the store. The `BaseStore` interface should support `search(query: str)`.
2.  **Injection:** The retrieved context needs to be formatted and appended to the system prompt.
3.  **Learning (Storage):**
    -   **Passive Learning:** Automatically store the User Query + Final Agent Response as a pair in the vector store. This acts as a "long-term memory" of past Q&A.
    -   **Active Learning:** The agent could have a special internal tool `save_knowledge(fact: str)` that it calls when it learns something new. This is more advanced but more precise.

## Implementation Plan

1.  **Create `agentflow/agent.py`:** Implement the `Agent` class.
2.  **Implement `_main_node`:** Add the logic for message conversion, `litellm` call, and response conversion.
3.  **Implement `learning` logic:**
    -   Integrate with `BaseStore`.
    -   Add retrieval logic before LLM call.
    -   Add simple "save interaction" logic after LLM call (or as a background task).
4.  **Update Examples:** Refactor `react-mcp` and `react_di` examples to use the new `Agent` class to demonstrate the reduction in code.


## Appendix E: Additional Resources

### Related Documentation
- [StateGraph Documentation](docs/concepts/graphs.md)
- [Node Documentation](docs/concepts/nodes.md)
- [ToolNode Documentation](docs/concepts/tools.md)
- [BaseStore Documentation](docs/concepts/stores.md)
- [LiteLLM Provider List](https://docs.litellm.ai/docs/providers)

### Community Examples
- [Example Repository](https://github.com/Iamsdt/PyAgenity/tree/main/examples)
- [Discord Community](https://discord.gg/agentflow)
- [Discussions Forum](https://github.com/Iamsdt/PyAgenity/discussions)

### Video Tutorials (Coming Soon)
1. Introduction to Agent Class
2. Building a Multi-Agent System
3. Implementing Learning with Agent
4. When to Use Agent vs Custom Nodes

---

*End of Document*
