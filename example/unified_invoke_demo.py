"""
Demo of unified invoke API that auto-detects fresh vs resume execution.
"""

import asyncio
from typing import Any

from pyagenity.graph.checkpointer import InMemoryCheckpointer
from pyagenity.graph.graph import StateGraph
from pyagenity.graph.state.agent_state import AgentState
from pyagenity.graph.utils import END, Message


def dummy_realtime_sync(
    state: AgentState, messages: list[Message], exec_meta: dict, config: dict[str, Any]
) -> None:
    """Dummy sync implementation for demonstration."""
    thread_id = config.get("thread_id", "default")
    print(
        f"[SYNC] Thread {thread_id}: Node {exec_meta.get('current_node')}, Step {exec_meta.get('step')}"
    )
    print(f"[SYNC] Messages count: {len(messages)}, State context: {len(state.context)}")


async def async_dummy_realtime_sync(
    state: AgentState, messages: list[Message], exec_meta: dict, config: dict[str, Any]
) -> None:
    """Dummy async implementation for demonstration."""
    thread_id = config.get("thread_id", "default")
    print(
        f"[ASYNC-SYNC] Thread {thread_id}: Node {exec_meta.get('current_node')}, Step {exec_meta.get('step')}"
    )
    await asyncio.sleep(0.01)  # Simulate async work


def step1_node(state: AgentState, config: dict[str, Any], checkpointer=None, store=None) -> str:
    print("Executing step1_node")
    if "resume_data" in config:
        print(f"Resume data provided: {config['resume_data']}")
    return "Step 1 completed"


def step2_node(state: AgentState, config: dict[str, Any], checkpointer=None, store=None) -> str:
    print("Executing step2_node")
    if "resume_data" in config:
        print(f"Resume data provided: {config['resume_data']}")
    return "Step 2 completed"


def step3_node(state: AgentState, config: dict[str, Any], checkpointer=None, store=None) -> str:
    print("Executing step3_node")
    if "resume_data" in config:
        print(f"Resume data provided: {config['resume_data']}")
    return "Step 3 completed"


def main():
    print("=== Unified Invoke API Demo ===")

    # Setup graph with interrupts
    graph = StateGraph()
    graph.add_node("step1", step1_node)
    graph.add_node("step2", step2_node)
    graph.add_node("step3", step3_node)

    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", END)

    graph.set_entry_point("step1")

    # Compile with interrupts and realtime sync
    checkpointer = InMemoryCheckpointer()
    compiled_graph = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["step1", "step2"],  # Interrupt after these nodes
        realtime_state_sync=dummy_realtime_sync,  # Use sync version
    )

    config = {"thread_id": "demo_user"}

    # First invoke - fresh execution (should pause after step1)
    print("\n--- First invoke (fresh execution) ---")
    result = compiled_graph.invoke(
        input_data={"messages": [Message.from_text("Start the process")]}, config=config
    )
    print(f"Result after first invoke: {result}")

    # Second invoke - auto-resume (should pause after step2)
    print("\n--- Second invoke (auto-resume) ---")
    result = compiled_graph.invoke(
        input_data={"additional_data": "some resume data"},  # This will be passed as resume_data
        config=config,
    )
    print(f"Result after second invoke: {result}")

    # Third invoke - auto-resume (should complete)
    print("\n--- Third invoke (auto-resume to completion) ---")
    result = compiled_graph.invoke(
        input_data={},  # No additional data needed
        config=config,
    )
    print(f"Final result: {result}")

    print("\n=== Testing with Async Sync Hook ===")

    # Test with async sync hook
    compiled_graph_async = graph.compile(
        checkpointer=InMemoryCheckpointer(),  # Fresh checkpointer
        interrupt_after=["step1"],  # Just one interrupt for simplicity
        realtime_state_sync=async_dummy_realtime_sync,  # Use async version
    )

    config_async = {"thread_id": "async_user"}

    # Fresh execution with async sync
    print("\n--- Async sync execution ---")
    result = compiled_graph_async.invoke(
        input_data={"messages": [Message.from_text("Start async process")]}, config=config_async
    )
    print(f"Result with async sync: {result}")

    # Resume execution
    print("\n--- Async sync resume ---")
    result = compiled_graph_async.invoke(input_data={}, config=config_async)
    print(f"Final async result: {result}")


if __name__ == "__main__":
    main()
