# Handoff Tool Implementation Plan

## Core Concept

**Handoff**: When an agent detects it needs to switch to another agent, it includes a handoff instruction in its message. We detect this during message parsing (before tool execution) and navigate to the target agent without executing the tool.

## Key Insight

Instead of making tools return Commands or creating complex logic, we use the **message content** to detect handoff intent:
- Agent returns a message with a special marker or structured content
- During message parsing in the graph, we detect the handoff
- We extract the target agent name and navigate there
- We DON'T add the handoff instruction to conversation history (clean)

## Implementation Strategy

### Step 1: Create HandoffBlock Content Type

**Location**: `agentflow/state/message_block.py`

**Add new dataclass**:
```python
@dataclass
class HandoffBlock(ContentBlock):
    """Represents a handoff to another agent."""
    type: Literal["handoff"] = "handoff"
    target_agent: str  # Name of the node to handoff to
    reason: str | None = None  # Why we're handing off
```

### Step 2: Create Handoff Detection Function

**Location**: `agentflow/prebuilt/tools/handoff.py`

**Functions**:
```python
def create_handoff_tool(
    *,
    agent_name: str,
    description: str | None = None,
) -> Callable:
    """
    Create a tool that initiates a handoff to another agent.
    
    When the LLM calls this tool, it returns a Message with a HandoffBlock.
    The graph execution will detect this and navigate to the target agent.
    
    Args:
        agent_name: Target agent node name
        description: Tool description for LLM
    
    Returns:
        Callable tool that when invoked, returns a Message with HandoffBlock
    
    Example:
        transfer_to_researcher = create_handoff_tool(
            agent_name="researcher",
            description="Transfer to research specialist when you need information"
        )
    """
    name = f"transfer_to_{agent_name}"
    desc = description or f"Transfer to {agent_name}"
    
    def handoff_tool(reason: str | None = None) -> Message:
        """Handoff to the target agent."""
        return Message(
            content=[
                HandoffBlock(
                    target_agent=agent_name,
                    reason=reason or f"Transferring to {agent_name}"
                )
            ],
            role="assistant",
        )
    
    handoff_tool.__name__ = name
    handoff_tool.__doc__ = desc
    
    return handoff_tool


def extract_handoff(message: Message) -> str | None:
    """
    Extract handoff target from a message if it contains a HandoffBlock.
    
    Args:
        message: Message to check
    
    Returns:
        Target agent name if handoff found, None otherwise
    """
    if not message.content:
        return None
    
    for block in message.content:
        if isinstance(block, HandoffBlock):
            return block.target_agent
    
    return None
```

### Step 3: Integrate into Message Parsing

**Location**: `agentflow/graph/utils/invoke_node_handler.py` or `stream_node_handler.py`

**In the node execution flow, after getting a message from the agent:**

```python
# After agent returns a message
if isinstance(result, Message):
    # Check if this is a handoff
    handoff_target = extract_handoff(result)
    
    if handoff_target:
        # This is a handoff, not a regular tool call
        # Create a Command to navigate
        return Command(
            update=None,  # Don't add the handoff message to history
            goto=handoff_target,
        )
    
    # Otherwise, continue with normal processing
    # Check for tool calls, etc.
```

### Step 4: Handle Handoff in Result Processing

**Location**: `agentflow/graph/utils/utils.py`

**Update `process_node_result()` to handle handoff returns:**

```python
def process_node_result(
    result: Any,
    state: StateT,
    messages: list[Message],
) -> tuple[StateT, list[Message], str | None]:
    """
    Process node result and extract next navigation.
    Now also handles handoff Messages.
    """
    next_node = None
    new_messages = []
    
    if isinstance(result, Message):
        # Check for handoff FIRST
        handoff_target = extract_handoff(result)
        if handoff_target:
            # Return handoff target as next node
            # Don't add message to history
            return state, messages, handoff_target
        
        # Normal message processing
        add_unique_message(result)
    
    # ... rest of processing
```

## Concrete Implementation

### File Changes Summary

```
1. agentflow/state/message_block.py
   - Add HandoffBlock dataclass

2. agentflow/prebuilt/tools/handoff.py (NEW)
   - create_handoff_tool()
   - extract_handoff()

3. agentflow/graph/utils/utils.py
   - Import HandoffBlock and extract_handoff
   - Update process_node_result() to check for handoff

4. agentflow/prebuilt/tools/__init__.py
   - Export create_handoff_tool

5. tests/graph/test_handoff.py (NEW)
   - Test basic handoff
   - Test multi-agent chains
```

## Usage Example

```python
from agentflow import StateGraph
from agentflow.prebuilt.tools import create_handoff_tool
from agentflow.graph.tool_node import ToolNode

# Create handoff tools
transfer_to_researcher = create_handoff_tool(
    agent_name="researcher",
    description="Transfer to research specialist for detailed investigation"
)

transfer_to_writer = create_handoff_tool(
    agent_name="writer", 
    description="Transfer to writing specialist for content creation"
)

# Add to agent's tools
coordinator_tools = ToolNode([
    transfer_to_researcher,
    transfer_to_writer,
    # ... other tools
])

# Build graph
graph = StateGraph()
graph.add_node("coordinator", coordinator_agent)
graph.add_node("coordinator_tools", coordinator_tools)
graph.add_node("researcher", researcher_agent)
graph.add_node("writer", writer_agent)

# When coordinator's LLM calls transfer_to_researcher:
# 1. Tool returns Message with HandoffBlock
# 2. Graph detects HandoffBlock during parsing
# 3. Navigation happens to "researcher" node
# 4. Handoff message NOT saved in history
```

## Advantages

✅ **Simple**: No tool execution needed, just message parsing
✅ **Clean**: Handoff message not in conversation history
✅ **Fast**: Detected before tool node execution
✅ **Explicit**: HandoffBlock is clear intent
✅ **No Decorator Needed**: Uses standard tool pattern
✅ **Flexible**: Can easily add reason/metadata

## Key Integration Points

1. **Message Detection**: During `process_node_result()` 
2. **Navigation**: Return Command with goto=target_agent
3. **History**: Don't add HandoffBlock messages to context
4. **Tool Format**: Tools are regular callables that return Messages

---

**Status**: Ready for Implementation
