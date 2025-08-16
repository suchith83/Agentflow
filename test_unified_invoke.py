"""
Simple test to verify unified invoke API step-by-step.
"""

from pyagenity.graph.checkpointer import InMemoryCheckpointer
from pyagenity.graph.graph import StateGraph
from pyagenity.graph.state.agent_state import AgentState
from pyagenity.graph.utils import END, Message


def step1_node(state: AgentState, config: dict, checkpointer=None, store=None) -> str:
    print(f"Step1: Current step = {state.execution_meta.step}")
    return "Step 1 completed"


def step2_node(state: AgentState, config: dict, checkpointer=None, store=None) -> str:
    print(f"Step2: Current step = {state.execution_meta.step}")
    return "Step 2 completed"


def main():
    # Setup simple graph
    graph = StateGraph()
    graph.add_node("step1", step1_node)
    graph.add_node("step2", step2_node)

    graph.add_edge("step1", "step2")
    graph.add_edge("step2", END)
    graph.set_entry_point("step1")

    # Compile with interrupt after step1
    checkpointer = InMemoryCheckpointer()
    compiled_graph = graph.compile(checkpointer=checkpointer, interrupt_after=["step1"])

    config = {"thread_id": "test_user"}

    print("=== Step 1: Fresh execution (should pause after step1) ===")
    result = compiled_graph.invoke(
        input_data={"messages": [Message.from_text("Start")]}, config=config
    )
    print(f"Messages count: {len(result[1])}")

    # Check state in checkpointer
    saved_state = checkpointer.get_state(config)
    print(f"Saved state interrupted: {saved_state.is_interrupted()}")
    print(f"Current node: {saved_state.execution_meta.current_node}")
    print(f"Step: {saved_state.execution_meta.step}")

    print("\n=== Step 2: Resume execution (should complete) ===")
    result = compiled_graph.invoke(input_data={}, config=config)
    print(f"Messages count: {len(result[1])}")

    # Check final state
    final_state = checkpointer.get_state(config)
    print(f"Final state interrupted: {final_state.is_interrupted()}")
    print(f"Final step: {final_state.execution_meta.step}")


if __name__ == "__main__":
    main()
