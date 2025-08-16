"""
Demo of pause/resume functionality in PyAgenity.

This example demonstrates:
1. Setting interrupt points during compilation
2. Pausing execution at specific nodes
3. Resuming execution with optional input
4. Multi-user state management
5. Checkpointer integration for state persistence
"""

import asyncio
import traceback
from typing import Any

from pyagenity.graph.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
from pyagenity.graph.graph.compiled_graph import CompiledGraph
from pyagenity.graph.graph.state_graph import StateGraph
from pyagenity.graph.state.agent_state import AgentState
from pyagenity.graph.utils.message import Message
from pyagenity.graph.utils.command import Command
from pyagenity.graph.utils.constants import END


async def user_input_node(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Node that waits for user input."""
    print(f"Current state step: {state.step}")

    # Check if we have resume data
    resume_data = config.get("resume_data")
    if resume_data:
        user_text = resume_data.get("user_input", "Resumed without input")
        print(f"Resuming with input: {user_text}")
    else:
        user_text = "Initial user input"
        print(f"Initial execution with: {user_text}")

    return Message.from_text(f"User said: {user_text}")


async def processing_node(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Node that processes the user input."""
    last_message = state.context[-1] if state.context else None
    processed_text = f"Processed: {last_message.content}" if last_message else "No input to process"

    print(f"Processing: {processed_text}")
    return Message.from_text(processed_text)


async def decision_node(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Command:
    """Node that makes a decision based on processed input."""
    last_message = state.context[-1] if state.context else None

    # Check if we have resume data that affects the decision
    resume_data = config.get("resume_data")
    if resume_data and "decision" in resume_data:
        decision = resume_data["decision"]
        print(f"Resuming with decision: {decision}")
        return Command(update=Message.from_text(f"Decision made: {decision}"), goto=decision)

    # Default decision based on content
    if last_message and "error" in last_message.content.lower():
        decision = "error_handler"
    else:
        decision = "success_handler"

    print(f"Decision: {decision}")
    return Command(update=Message.from_text(f"Decision made: {decision}"), goto=decision)


async def success_handler(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Handle successful processing."""
    return Message.from_text("Success! Processing completed successfully.")


async def error_handler(
    state: AgentState, config: dict[str, Any], checkpointer=None, store=None
) -> Message:
    """Handle error cases."""
    return Message.from_text("Error detected and handled appropriately.")


async def main():
    """Demonstrate pause/resume functionality."""
    print("=== PyAgenity Pause/Resume Demo ===\n")

    # Create graph with nodes
    graph = StateGraph()
    graph.add_node("input", user_input_node)
    graph.add_node("process", processing_node)
    graph.add_node("decide", decision_node)
    graph.add_node("success_handler", success_handler)
    graph.add_node("error_handler", error_handler)

    # Add edges
    graph.add_edge("input", "process")
    graph.add_edge("process", "decide")
    # decision_node uses Command.goto for routing, but we need fallback edges
    graph.add_edge("success_handler", END)  # Connect to END
    graph.add_edge("error_handler", END)  # Connect to END

    graph.set_entry_point("input")

    # Create checkpointer for state persistence
    checkpointer = InMemoryCheckpointer()

    print("1. Compiling graph with interrupt points...")
    # Compile with interrupt points
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["process", "decide"],  # Pause before these nodes
        interrupt_after=["input"],  # Pause after this node
    )

    # Initial state
    initial_state = AgentState()
    config = {"thread_id": "user_123"}

    print("2. Starting initial execution...")
    # First execution - should pause after 'input' node
    try:
        result_state, messages = await compiled.ainvoke(initial_state, config=config)
        print(f"   Execution paused. Messages: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"   Message {i + 1}: {msg.content}")
    except Exception as e:
        print(f"   Error during initial execution: {e}")
        traceback.print_exc()
        return

    print("\n3. Resuming execution (should pause before 'process')...")
    # Resume - should pause before 'process' node
    try:
        result_state, messages = await compiled.aresume(config=config)
        print(f"   Execution paused again. Messages: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"   Message {i + 1}: {msg.content}")
    except Exception as e:
        print(f"   Error during first resume: {e}")
        return

    print("\n4. Resuming with additional input...")
    # Resume with input that affects processing
    resume_input = {"user_input": "Please process this special request"}
    try:
        result_state, messages = await compiled.aresume(input_data=resume_input, config=config)
        print(f"   Execution paused again. Messages: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"   Message {i + 1}: {msg.content}")
    except Exception as e:
        print(f"   Error during second resume: {e}")
        return

    print("\n5. Final resume with decision override...")
    # Resume with input that overrides the decision
    decision_input = {"decision": "success_handler"}
    try:
        result_state, messages = await compiled.aresume(input_data=decision_input, config=config)
        print(f"   Execution completed. Messages: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"   Message {i + 1}: {msg.content}")
    except Exception as e:
        print(f"   Error during final resume: {e}")
        return

    print("\n6. Demonstrating multi-user isolation...")
    # Start a different user's execution
    user2_config = {"thread_id": "user_456"}
    user2_state = AgentState()

    try:
        user2_result, user2_messages = await compiled.ainvoke(user2_state, config=user2_config)
        print(f"   User 2 paused independently. Messages: {len(user2_messages)}")

        # Resume user 2 directly to completion
        while True:
            try:
                user2_result, new_messages = await compiled.aresume(config=user2_config)
                user2_messages.extend(new_messages)
                print(f"   User 2 resume step - Messages: {len(new_messages)}")
            except Exception:
                # If resume fails, execution is complete
                break

        print(f"   User 2 completed with {len(user2_messages)} total messages")

    except Exception as e:
        print(f"   Error in multi-user demo: {e}")

    print("\n=== Demo Complete ===")
    print("This demo showed:")
    print("- Setting interrupt points during graph compilation")
    print("- Pausing execution at interrupt_before and interrupt_after points")
    print("- Resuming with optional input data that affects node behavior")
    print("- Multi-user state isolation (different users paused at different points)")
    print("- Checkpointer integration for state persistence")


if __name__ == "__main__":
    asyncio.run(main())
