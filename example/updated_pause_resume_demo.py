"""
Updated pause/resume demo with new API:
- No direct AgentState creation
- Only dict inputs to invoke/resume
- Dummy realtime_state_sync hook (sync and async variants)
- State persisted by thread_id in checkpointer
"""

from typing import Any

from pyagenity.graph.checkpointer import InMemoryCheckpointer
from pyagenity.graph.graph import StateGraph
from pyagenity.graph.state.agent_state import AgentState
from pyagenity.graph.utils import END, Message


# Demo node functions (updated to accept 4 parameters)
def user_input_node(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> AgentState:
    """Node that handles user input."""
    print("📝 User Input Node executed")

    # Check for resume data
    resume_data = config.get("resume_data")
    if resume_data:
        print(f"  📤 Resume data received: {resume_data}")
        # Add resume data as a new message
        new_msg = Message.from_text(f"Resume input: {resume_data.get('input', 'N/A')}")
        state.context.append(new_msg)

    # Add a message to show this node ran
    state.context.append(Message.from_text("User input processed"))
    state.step += 1
    return state


def process_step1(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> AgentState:
    """Step 1 processing node."""
    print("🔄 Step 1 Node executed")
    state.context.append(Message.from_text("Step 1 completed"))
    state.step += 1
    return state


def process_step2(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> AgentState:
    """Step 2 processing node."""
    print("🔄 Step 2 Node executed")
    state.context.append(Message.from_text("Step 2 completed"))
    state.step += 1
    return state


def final_response_node(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> AgentState:
    """Final response node."""
    print("✅ Final Response Node executed")
    state.context.append(Message.from_text("Final response generated"))
    state.step += 1
    return state


def end_node(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> AgentState:
    """End node."""
    print("🏁 End Node executed")
    state.context.append(Message.from_text("Process completed"))
    return state


# Dummy realtime sync hooks (sync and async variants)
def dummy_realtime_sync(
    state: AgentState, messages: list[Message], exec_meta: dict[str, Any], config: dict[str, Any]
) -> None:
    """Dummy synchronous realtime state sync hook.

    In a real implementation, this might write to Redis or a cache.
    """
    thread_id = config.get("thread_id", "default")
    print(
        f"  🔄 [SYNC] Realtime sync for thread {thread_id}: "
        f"step={exec_meta.get('step', 0)}, "
        f"node={exec_meta.get('current_node', 'unknown')}, "
        f"messages={len(messages)}"
    )


async def dummy_async_realtime_sync(
    state: AgentState, messages: list[Message], exec_meta: dict[str, Any], config: dict[str, Any]
) -> None:
    """Dummy async realtime state sync hook.

    In a real implementation, this might write to Redis or a cache asynchronously.
    """
    thread_id = config.get("thread_id", "default")
    print(
        f"  🔄 [ASYNC] Realtime sync for thread {thread_id}: "
        f"step={exec_meta.get('step', 0)}, "
        f"node={exec_meta.get('current_node', 'unknown')}, "
        f"messages={len(messages)}"
    )


def test_basic_pause_resume():
    """Test basic pause/resume with sync realtime hook."""
    print("\n" + "=" * 60)
    print("🧪 Testing Basic Pause/Resume with Sync Realtime Hook")
    print("=" * 60)

    # Create checkpointer
    checkpointer = InMemoryCheckpointer()

    # Create graph
    graph = StateGraph()
    graph.add_node("user_input", user_input_node)
    graph.add_node("step1", process_step1)
    graph.add_node("step2", process_step2)
    graph.add_node("final", final_response_node)

    # Set up edges
    graph.set_entry_point("user_input")
    graph.add_edge("user_input", "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "final")
    graph.add_edge("final", END)

    # Compile with interrupts and realtime sync
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["step1", "step2"],
        realtime_state_sync=dummy_realtime_sync,  # Sync hook
        debug=True,
    )

    # Initial input (dict only)
    input_data = {"messages": [Message.from_text("Hello, start the process")]}
    config = {"thread_id": "user1"}

    print("\n📤 Starting execution...")
    # First run - should pause after step1
    result1 = compiled.invoke(input_data, config)
    print(f"📥 First result: {len(result1['messages'])} messages")

    print("\n📤 Resuming execution...")
    # Resume - should pause after step2
    result2 = compiled.resume(config=config)
    print(f"📥 Second result: {len(result2['messages'])} messages")

    print("\n📤 Final resume...")
    # Final resume - should complete
    result3 = compiled.resume(config=config)
    print(f"📥 Final result: {len(result3['messages'])} messages")

    print("✅ Basic pause/resume test completed!")


def test_async_realtime_hook():
    """Test pause/resume with async realtime hook."""
    print("\n" + "=" * 60)
    print("🧪 Testing Pause/Resume with Async Realtime Hook")
    print("=" * 60)

    # Create checkpointer
    checkpointer = InMemoryCheckpointer()

    # Create graph
    graph = StateGraph()
    graph.add_node("user_input", user_input_node)
    graph.add_node("step1", process_step1)
    graph.add_node("final", final_response_node)

    # Set up edges
    graph.set_entry_point("user_input")
    graph.add_edge("user_input", "step1")
    graph.add_edge("step1", "final")
    graph.add_edge("final", END)

    # Compile with async realtime sync
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["step1"],
        realtime_state_sync=dummy_async_realtime_sync,  # Async hook
        debug=True,
    )

    # Initial input
    input_data = {"messages": [Message.from_text("Hello async world")]}
    config = {"thread_id": "user2"}

    print("\n📤 Starting async execution...")
    result1 = compiled.invoke(input_data, config)
    print(f"📥 First result: {len(result1['messages'])} messages")

    print("\n📤 Resuming async execution...")
    result2 = compiled.resume(config=config)
    print(f"📥 Final result: {len(result2['messages'])} messages")

    print("✅ Async realtime hook test completed!")


def test_resume_with_input():
    """Test resume with additional input data."""
    print("\n" + "=" * 60)
    print("🧪 Testing Resume with Input Data")
    print("=" * 60)

    checkpointer = InMemoryCheckpointer()

    # Simple graph
    graph = StateGraph()
    graph.add_node("user_input", user_input_node)
    graph.add_node("final", final_response_node)

    graph.set_entry_point("user_input")
    graph.add_edge("user_input", "final")
    graph.add_edge("final", END)

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["user_input"],
        realtime_state_sync=dummy_realtime_sync,
        debug=True,
    )

    # Initial execution
    input_data = {"messages": [Message.from_text("Start process")]}
    config = {"thread_id": "user3"}

    print("\n📤 Starting execution...")
    result1 = compiled.invoke(input_data, config)

    # Resume with additional input
    resume_input = {"input": "additional user data from second interaction"}
    print(f"\n📤 Resuming with input: {resume_input}")
    result2 = compiled.resume(input_data=resume_input, config=config)

    print("✅ Resume with input test completed!")


def test_multi_user():
    """Test multi-user isolation."""
    print("\n" + "=" * 60)
    print("🧪 Testing Multi-User State Isolation")
    print("=" * 60)

    checkpointer = InMemoryCheckpointer()

    # Create graph
    graph = StateGraph()
    graph.add_node("step1", process_step1)
    graph.add_node("step2", process_step2)

    graph.set_entry_point("step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", END)

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["step1"],  # Both users will be interrupted after step1
        realtime_state_sync=dummy_realtime_sync,
        debug=True,
    )

    # User 1 starts execution
    user1_input = {"messages": [Message.from_text("User 1 request")]}
    user1_config = {"thread_id": "user1"}

    print("\n👤 User 1 starting execution...")
    result1 = compiled.invoke(user1_input, user1_config)

    # User 2 starts execution (independent)
    user2_input = {"messages": [Message.from_text("User 2 request")]}
    user2_config = {"thread_id": "user2"}

    print("\n👤 User 2 starting execution...")
    result2 = compiled.invoke(user2_input, user2_config)

    # User 1 resumes
    print("\n👤 User 1 resuming...")
    result1_final = compiled.resume(config=user1_config)

    # User 2 resumes
    print("\n👤 User 2 resuming...")
    result2_final = compiled.resume(config=user2_config)

    print("✅ Multi-user isolation test completed!")


if __name__ == "__main__":
    # Run all tests
    test_basic_pause_resume()
    test_async_realtime_hook()
    test_resume_with_input()
    test_multi_user()

    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("=" * 60)
