"""
Updated demo of unified invoke API (auto-detects fresh vs resume).

This example demonstrates:
1. Unified invoke() method that auto-detects fresh vs resume execution
2. Automatic pause/resume based on interrupt configuration
3. Multi-user state management with thread isolation
4. Realtime state sync hooks (dummy implementation)
"""

import asyncio
from typing import Any

from pyagenity.graph.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from pyagenity.graph.graph.state_graph import StateGraph
from pyagenity.graph.state.agent_state import AgentState
from pyagenity.graph.utils.message import Message
from pyagenity.graph.utils.command import Command
from pyagenity.graph.utils.constants import END


def dummy_realtime_sync(
    state: AgentState, messages: list[Message], exec_meta: dict, config: dict[str, Any]
) -> None:
    """Dummy realtime sync for demonstration."""
    thread_id = config.get("thread_id", "default")
    current_node = exec_meta.get("current_node", "unknown")
    step = exec_meta.get("step", 0)
    print(
        f"  [SYNC] Thread {thread_id}: Step {step} at node '{current_node}' - {len(messages)} new messages"
    )


async def user_input_node(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Node that handles user input."""
    print(f"  Executing user_input_node (step {state.execution_meta.step})")

    # Check if we have resume data
    resume_data = config.get("resume_data")
    if resume_data:
        user_text = resume_data.get("user_input", "Resumed without specific input")
        print(f"    Resume data: {user_text}")
    else:
        user_text = "Initial user input"
        print(f"    Fresh execution: {user_text}")

    return Message.from_text(f"User said: {user_text}")


async def processing_node(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Node that processes the user input."""
    print(f"  Executing processing_node (step {state.execution_meta.step})")

    last_message = state.context[-1] if state.context else None
    processed_text = f"Processed: {last_message.content}" if last_message else "No input to process"

    print(f"    Processing: {processed_text}")
    return Message.from_text(processed_text)


async def decision_node(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Command:
    """Node that makes a decision based on processed input."""
    print(f"  Executing decision_node (step {state.execution_meta.step})")

    last_message = state.context[-1] if state.context else None

    # Check if we have resume data that affects the decision
    resume_data = config.get("resume_data")
    if resume_data and "decision" in resume_data:
        decision = resume_data["decision"]
        print(f"    Resuming with decision override: {decision}")
        return Command(update=Message.from_text(f"Decision made: {decision}"), goto=decision)

    # Default decision based on content
    if last_message and "error" in last_message.content.lower():
        decision = "error_handler"
    else:
        decision = "success_handler"

    print(f"    Decision: {decision}")
    return Command(update=Message.from_text(f"Decision made: {decision}"), goto=decision)


async def success_handler(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Handle successful processing."""
    print(f"  Executing success_handler (step {state.execution_meta.step})")
    return Message.from_text("Success! Processing completed successfully.")


async def error_handler(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Handle error cases."""
    print(f"  Executing error_handler (step {state.execution_meta.step})")
    return Message.from_text("Error detected and handled appropriately.")


async def main():
    """Demonstrate unified invoke API."""
    print("=== PyAgenity Unified Invoke API Demo ===\\n")

    # Create graph with nodes
    graph = StateGraph()
    graph.add_node("input", user_input_node)
    graph.add_node("process", processing_node)
    graph.add_node("decide", decision_node)
    graph.add_node("success_handler", success_handler)
    graph.add_node("error_handler", error_handler)

    # Set up edges
    graph.add_edge("input", "process")
    graph.add_edge("process", "decide")
    graph.add_edge("success_handler", END)
    graph.add_edge("error_handler", END)

    graph.set_entry_point("input")

    # Compile with interrupts and realtime sync
    checkpointer = InMemoryCheckpointer()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_after=["input", "process"],  # Pause after these nodes
        realtime_state_sync=dummy_realtime_sync,
    )

    config = {"thread_id": "user_123"}

    print("1. Starting fresh execution (should pause after 'input')...")
    try:
        result = await compiled.ainvoke(
            input_data={"messages": [Message.from_text("Start the workflow")]}, config=config
        )
        # Parse result to get messages
        messages = result.get("messages", [])
        print(f"   Result: {len(messages)} messages")
        print(f"   Last message: {messages[-1].content if messages else 'None'}")
    except Exception as e:
        print(f"   Error: {e}")
        return

    print("\\n2. Auto-resuming (should pause after 'process')...")
    try:
        result = await compiled.ainvoke(
            input_data={},  # No additional data, just resume
            config=config,
        )
        messages = result.get("messages", [])
        print(f"   Result: {len(messages)} messages")
        print(f"   Last message: {messages[-1].content if messages else 'None'}")
    except Exception as e:
        print(f"   Error: {e}")
        return

    print("\\n3. Auto-resuming with decision override...")
    try:
        result = await compiled.ainvoke(
            input_data={"decision": "success_handler"},  # Override decision
            config=config,
        )
        messages = result.get("messages", [])
        print(f"   Result: {len(messages)} messages")
        print(f"   Last message: {messages[-1].content if messages else 'None'}")
    except Exception as e:
        print(f"   Error: {e}")
        return

    print("\\n4. Demonstrating multi-user isolation...")
    # Start a different user's execution
    user2_config = {"thread_id": "user_456"}

    print("   Starting User 2...")
    try:
        result = await compiled.ainvoke(
            input_data={"messages": [Message.from_text("User 2 workflow")]}, config=user2_config
        )
        messages = result.get("messages", [])
        print(f"   User 2 paused: {len(messages)} messages")

        # Resume User 2 with error to test error_handler path
        print("   Resuming User 2 with error simulation...")
        result = await compiled.ainvoke(
            input_data={"user_input": "This has an error in it"}, config=user2_config
        )
        messages = result.get("messages", [])
        print(f"   User 2 paused again: {len(messages)} messages")

        # Complete User 2 execution
        print("   Completing User 2...")
        result = await compiled.ainvoke(input_data={}, config=user2_config)
        messages = result.get("messages", [])
        print(f"   User 2 completed: {len(messages)} messages")
        print(f"   Final message: {messages[-1].content if messages else 'None'}")

    except Exception as e:
        print(f"   User 2 error: {e}")

    print("\\n=== Demo Complete ===")
    print("This demo showed:")
    print("- Unified invoke() API that auto-detects fresh vs resume")
    print("- Automatic state persistence and restoration")
    print("- Multi-user isolation with separate thread_ids")
    print("- Resume with additional input data")
    print("- Realtime state sync hooks")


if __name__ == "__main__":
    asyncio.run(main())
