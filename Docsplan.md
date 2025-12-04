# Documentation Rewrite Plan - Version 0.5.3+

## Overview

With the introduction of the **Agent class** in version 0.5.3, we're shifting the documentation narrative to promote a "**Simple but Powerful**" approach. The Agent class provides a streamlined way to build agents without boilerplate, while the StateGraph + custom functions approach remains available for advanced use cases.

---

## Key Messaging Strategy

### Primary Message
> **"Start simple with Agent class. Go custom when you need control."**

### Marketing Angle
- **Simple but Complex**: The Agent class handles 90% of use cases with minimal code
- **Zero Boilerplate**: No need to manually convert messages, handle tool logic, or manage LLM calls
- **Full Power When Needed**: Seamlessly transition to custom functions for advanced scenarios
- **Same Graph Engine**: Agent class uses the same StateGraph under the hood

---

## Documentation Structure Changes

### 1. Homepage (`docs/index.md`) Updates

**Priority: HIGH** ✅ COMPLETED

Changes needed:
- [x] Update Quick Start section to use Agent class as the primary example
- [x] Add "Two Ways to Build" section explaining Agent class vs custom functions
- [x] Update "Simple Example" to show Agent class first, then custom approach
- [x] Add comparison table: Agent class vs Custom functions
- [x] Update feature highlights to mention Agent class simplicity

### 2. New Tutorial: Agent Class (`docs/Tutorial/agent-class.md`)

**Priority: HIGH** ✅ COMPLETED

Create comprehensive guide covering:
- [x] Introduction: Why Agent class exists
- [x] Quick start with Agent class (5-minute guide)
- [x] Agent class parameters explained
- [x] Adding tools to Agent
- [x] Using tool_node_name for external ToolNodes
- [x] Context trimming with trim_context
- [x] Tool filtering with tags
- [x] Custom system prompts
- [x] LLM kwargs (temperature, max_tokens, etc.)
- [x] When to use Agent class vs custom functions
- [x] Migration guide: Converting custom functions to Agent class

### 3. Tutorial Index Update (`docs/Tutorial/index.md`)

**Priority: HIGH** ✅ COMPLETED

Changes needed:
- [x] Add Agent class tutorial as first item in tutorial path
- [x] Update learning objectives to mention Agent class
- [x] Add "Choose Your Path" section:
  - Quick Path: Agent class tutorials
  - Advanced Path: Custom function tutorials
- [x] Update table with Agent class entry

### 4. React Tutorial Updates (`docs/Tutorial/react/`)

**Priority: MEDIUM** ✅ COMPLETED

Changes needed:
- [x] Add new tutorial `00-agent-class-react.md` as the first entry
- [x] Update `README.md` to mention Agent class approach first
- [x] Add notes to existing tutorials about Agent class alternative
- [x] Update tutorial order in navigation

### 5. Concept Documentation (`docs/Concept/graph/`)

**Priority: MEDIUM** ✅ COMPLETED

Changes needed:
- [x] Add `agent-class.md` explaining the Agent abstraction
- [x] Update existing docs to reference Agent class where relevant
- [x] Add section on "Choosing the Right Approach"

### 6. Main README.md Update

**Priority: HIGH** ✅ COMPLETED

If README.md exists and is different from docs/index.md:
- [x] Update with Agent class as primary approach
- [x] Add quick comparison section

---

## Detailed Task Breakdown

### Phase 1: Core Documentation (Do First)

#### Task 1.1: Update Homepage Quick Start
- Replace complex example with Agent class example
- Keep custom function example as "Advanced Alternative"
- Add clear messaging about when to use each approach

#### Task 1.2: Create Agent Class Tutorial
- Comprehensive guide with examples
- Cover all Agent class parameters
- Include real-world patterns
- Add troubleshooting section

#### Task 1.3: Update Tutorial Index
- Restructure learning path
- Add Agent class as recommended starting point

### Phase 2: React Tutorial Updates

#### Task 2.1: Create Agent Class React Tutorial
- Show ReAct pattern with Agent class
- Compare with traditional approach
- Highlight reduced boilerplate

#### Task 2.2: Update React README
- New navigation with Agent class first
- Updated descriptions

### Phase 3: Concept Documentation

#### Task 3.1: Create Agent Class Concept Page
- Architecture explanation
- Internal workings
- Extension points

#### Task 3.2: Update Related Concept Pages
- Cross-reference Agent class
- Add migration tips

---

## Code Examples to Include

### Minimal Agent Class Example (Primary)
```python
from agentflow.graph import Agent, StateGraph, ToolNode
from agentflow.state import AgentState, Message
from agentflow.utils.constants import END

# Define your tool
def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 72°F"

# Create the graph with Agent class
graph = StateGraph()
graph.add_node("MAIN", Agent(
    model="gemini/gemini-2.5-flash",
    system_prompt=[{"role": "system", "content": "You are a helpful assistant."}],
    tool_node_name="TOOL"
))
graph.add_node("TOOL", ToolNode([get_weather]))

# Simple routing
def route(state: AgentState) -> str:
    if state.context and state.context[-1].tools_calls:
        return "TOOL"
    return END

graph.add_conditional_edges("MAIN", route, {"TOOL": "TOOL", END: END})
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile()
result = app.invoke({"messages": [Message.text_message("What's the weather in NYC?")]})
```

### Custom Function Example (Advanced)
```python
# Show the traditional approach with full control
# ... existing complex example ...
```

---

## Comparison Table Content

| Feature | Agent Class | Custom Functions |
|---------|-------------|------------------|
| Setup complexity | Minimal | Full control |
| Lines of code | ~10-20 | ~50-100 |
| Message conversion | Automatic | Manual |
| Tool handling | Built-in | Manual |
| LLM calls | Automatic | Manual |
| Streaming | Supported | Manual setup |
| Learning/RAG | Built-in option | Custom implementation |
| Context trimming | Built-in | Custom implementation |
| Best for | Most use cases | Complex workflows |

---

## Files to Create/Modify

### New Files
1. `docs/Tutorial/agent-class.md` - Main Agent class tutorial
2. `docs/Tutorial/react/00-agent-class-react.md` - Agent class React tutorial
3. `docs/Concept/graph/agent-class.md` - Agent class concept documentation

### Files to Modify
1. `docs/index.md` - Homepage with new Quick Start
2. `docs/Tutorial/index.md` - Updated tutorial path
3. `docs/Tutorial/react/README.md` - Updated React tutorial index
4. `mkdocs.yml` - Navigation updates (if needed)

---

## Implementation Order

1. **`docs/Tutorial/agent-class.md`** - Create comprehensive Agent class tutorial
2. **`docs/index.md`** - Update homepage with Agent class as primary approach
3. **`docs/Tutorial/index.md`** - Update tutorial navigation
4. **`docs/Tutorial/react/00-agent-class-react.md`** - Agent class React tutorial
5. **`docs/Tutorial/react/README.md`** - Update React tutorial index
6. **`docs/Concept/graph/agent-class.md`** - Concept documentation
7. **`mkdocs.yml`** - Update navigation if needed

---

## Success Criteria

- [ ] New users immediately see Agent class as the recommended starting point
- [ ] Clear guidance on when to use Agent class vs custom functions
- [ ] All Agent class parameters are documented with examples
- [ ] Existing documentation is updated with Agent class references
- [ ] Code examples are complete and runnable
- [ ] Migration path is clear for users upgrading from custom functions

---

## Timeline Estimate

- Phase 1 (Core): 2-3 hours
- Phase 2 (React): 1-2 hours  
- Phase 3 (Concepts): 1-2 hours
- Total: 4-7 hours
