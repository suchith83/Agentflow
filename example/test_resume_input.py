"""
Test resume with input functionality.
"""

import asyncio
from typing import Any

from pyagenity.graph.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from pyagenity.graph.graph.state_graph import StateGraph
from pyagenity.graph.state.agent_state import AgentState
from pyagenity.graph.utils.message import Message
from pyagenity.graph.utils.constants import END


async def input_sensitive_node(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Node that responds to resume input."""
    resume_data = config.get("resume_data")
    if resume_data and "user_message" in resume_data:
        msg = f"Received resume input: {resume_data['user_message']}"
        print(f"📨 {msg}")
        return Message.from_text(msg)

    msg = "No resume input received"
    print(f"📭 {msg}")
    return Message.from_text(msg)


async def main():
    """Test resume with input."""
    print("=== Resume with Input Test ===")

    # Create simple graph
    graph = StateGraph()
    graph.add_node("input_node", input_sensitive_node)
    graph.add_edge("input_node", END)
    graph.set_entry_point("input_node")

    checkpointer = InMemoryCheckpointer()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["input_node"],  # Pause after execution
    )

    config = {"thread_id": "test"}
    initial_state = AgentState()

    # Step 1: Execute and pause
    print("1. Initial execution...")
    result, messages = await compiled.ainvoke(initial_state, config=config)
    print(f"   Messages: {[m.content for m in messages]}")

    # Step 2: Resume with input
    print("\n2. Resume with input...")
    resume_input = {"user_message": "Hello from resume!"}
    result, messages = await compiled.aresume(input_data=resume_input, config=config)
    print(f"   Messages: {[m.content for m in messages]}")

    print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
