import pytest

from pyagenity.graph.tool_node import ToolNode
from pyagenity.prebuilt.agent import DeepResearchAgent
from pyagenity.state import AgentState
from pyagenity.utils import Message


def _assistant_with_tool_call(name: str, args: dict | None = None):
    return Message.create(
        role="assistant",
        content="use tool",
        tools_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": __import__("json").dumps(args or {})},
            }
        ],
    )


def test_compile_and_basic_flow():
    def plan(state: AgentState, config: dict):
        # First, propose a tool call
        return _assistant_with_tool_call("search", {"q": "hello"})

    def synthesize(state: AgentState, config: dict):
        return Message.create(role="assistant", content="synthesized")

    def critique(state: AgentState, config: dict):
        # Finish immediately
        return Message.create(role="assistant", content="done")

    def search(q: str) -> str:
        return f"result:{q}"

    tools = ToolNode([search])

    agent = DeepResearchAgent(max_iters=1, heavy_mode=False)
    app = agent.compile(
        plan_node=plan,
        research_tool_node=tools,
        synthesize_node=synthesize,
        critique_node=critique,
    )

    res = app.invoke(
        {"messages": [Message.from_text("do research")]},
        config={"thread_id": "test_dr_basic"},
    )
    assert "messages" in res  # noqa: S101
    # Should contain the tool result and assistant responses
    contents = [m.content for m in res["messages"]]
    assert any("result:hello" in c for c in contents)  # noqa: S101


@pytest.mark.asyncio
async def test_iteration_cap_and_heavy_mode():
    # This plan always requests tools
    def plan(state: AgentState, config: dict):
        return _assistant_with_tool_call("search", {"q": f"step{state.execution_meta.step}"})

    # Synth just echoes
    def synthesize(state: AgentState, config: dict):
        return Message.create(role="assistant", content="synth")

    # Critique will also request another tool call so router may loop
    def critique(state: AgentState, config: dict):
        return _assistant_with_tool_call("search", {"q": "next"})

    def search(q: str) -> str:
        return f"r:{q}"

    tools = ToolNode([search])

    # max_iters=1 should stop after at most one CRITIQUE->RESEARCH loop
    agent = DeepResearchAgent(max_iters=1, heavy_mode=True)
    app = agent.compile(
        plan_node=plan,
        research_tool_node=tools,
        synthesize_node=synthesize,
        critique_node=critique,
    )

    out = await app.ainvoke(
        {"messages": [Message.from_text("start")]},
        config={"thread_id": "test_dr_async"},
    )
    msgs = out.get("messages", [])
    # Ensure we ended and did not loop indefinitely
    assert isinstance(msgs, list)  # noqa: S101
    # Provide a sanity check that at least one search result exists
    assert any(m.role == "tool" for m in msgs)  # noqa: S101
