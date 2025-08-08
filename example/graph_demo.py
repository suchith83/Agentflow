import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pyagenity.agent.agent import Agent
from pyagenity.graph import (
    Graph,
    Edge,
    LLMNode,
    FunctionNode,
    HumanInputNode,
    GraphExecutor,
    SessionStatus,
)

# Simple prompt builders


def summarize_builder(state):
    user_input = state.get("last_user_input", "")
    return f"Summarize the following text in one sentence:\n{user_input}"  # noqa


def critique_builder(state):
    prev = state.get("last_response", "")
    return f"Provide a brief critique of this summary and suggest improvement:\n{prev}"  # noqa


def improvement_func(state):
    summary = state.get("last_response", "")
    critique = state.get("critique_response", "")
    state["final_suggestion"] = (
        f"Improved Summary Proposal based on critique: {critique[:100]} | Original: {summary[:100]}"
    )
    return state


def main():
    # Create agents (models are placeholders; configure via environment for litellm)
    summarize_agent = Agent(
        name="summarizer", model="gemini-2.0-flash", custom_llm_provider="gemini"
    )
    critique_agent = Agent(
        name="critic", model="gemini-2.0-flash", custom_llm_provider="gemini"
    )

    g = Graph()
    g.add_node(
        LLMNode(
            name="summarize", agent=summarize_agent, prompt_builder=summarize_builder
        ),
        start=True,
    )
    g.add_node(
        LLMNode(
            name="critique",
            agent=critique_agent,
            prompt_builder=critique_builder,
            output_key="critique_response",
        )
    )
    g.add_node(FunctionNode(name="improve", func=improvement_func))
    g.add_node(HumanInputNode(name="human_review"))

    # Edges
    g.add_edge(Edge("summarize", "critique"))
    g.add_edge(Edge("critique", "human_review"))
    g.add_edge(
        Edge(
            "human_review",
            "improve",
            condition=lambda s: s.get("human_input") is not None,
        )
    )

    executor = GraphExecutor(g)

    # Final hook
    def final_hook(state):
        print("--- FINAL STATE ---")
        print("Status:", state.status)
        print("Shared keys:", list(state.shared.keys()))
        if state.status == SessionStatus.COMPLETED:
            print("Final suggestion:", state.shared.get("final_suggestion"))

    executor.add_final_hook(final_hook)

    # Start graph
    text = "PyAgenity is a lightweight agent framework supporting multi-agent graphs with pause/resume."
    result = executor.start({"last_user_input": text})
    print(
        "Session:",
        result.session_id,
        "Status:",
        result.status,
        "Node:",
        result.current_node,
    )

    if result.status == SessionStatus.WAITING_HUMAN:
        # Simulate human input
        human = "Looks good, proceed with improvement."
        resumed = executor.resume(result.session_id, human_input=human)
        print("Resumed Status:", resumed.status)


if __name__ == "__main__":
    main()
