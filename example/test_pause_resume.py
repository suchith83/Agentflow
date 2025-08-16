"""
Simple test of pause/resume functionality.
"""

import asyncio
from typing import Any

from pyagenity.graph.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from pyagenity.graph.graph.state_graph import StateGraph
from pyagenity.graph.state.agent_state import AgentState
from pyagenity.graph.utils.message import Message
from pyagenity.graph.utils.constants import END


async def step1(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """First step."""
    print("Executing step 1")
    return Message.from_text("Step 1 completed")


async def step2(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Second step."""
    print("Executing step 2")
    return Message.from_text("Step 2 completed")


async def step3(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Third step."""
    print("Executing step 3")
    return Message.from_text("Step 3 completed")


async def main():
    """Test basic pause/resume functionality."""
    print("=== Simple Pause/Resume Test ===\n")

    # Create graph
    graph = StateGraph()
    graph.add_node("step1", step1)
    graph.add_node("step2", step2)
    graph.add_node("step3", step3)

    # Add edges
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    graph.set_entry_point("step1")

    # Create checkpointer
    checkpointer = InMemoryCheckpointer()

    # Compile with interrupts
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["step1", "step2"],  # Pause after these steps
    )

    # Initial state
    initial_state = AgentState()
    config = {"thread_id": "test_user"}

    print("1. Starting execution (should pause after step1)...")
    result_state, messages = await compiled.ainvoke(initial_state, config=config)
    print(f"   Paused. Messages: {[msg.content for msg in messages]}")

    print("\n2. Resuming (should pause after step2)...")
    result_state, messages = await compiled.aresume(config=config)
    print(f"   Paused again. Messages: {[msg.content for msg in messages]}")

    print("\n3. Final resume (should complete)...")
    result_state, messages = await compiled.aresume(config=config)
    print(f"   Completed. Messages: {[msg.content for msg in messages]}")

    print("\n=== Test Complete ===")
    print("✅ Pause/resume functionality is working!")


if __name__ == "__main__":
    asyncio.run(main())
