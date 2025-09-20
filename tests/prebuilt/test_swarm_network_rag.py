from pyagenity.prebuilt.agent import NetworkAgent, RAGAgent, SwarmAgent
from pyagenity.state import AgentState
from pyagenity.utils import Message


def _assistant_msg(text: str) -> Message:
    return Message.create(role="assistant", content=text)


def test_swarm_agent_basic_flow():
    # Define simple worker, collect, and consensus nodes
    def worker_a(state: AgentState, config: dict):
        return _assistant_msg("worker_a")

    def worker_b(state: AgentState, config: dict):
        return _assistant_msg("worker_b")

    def collect(state: AgentState, config: dict):
        return _assistant_msg("collect")

    def consensus(state: AgentState, config: dict):
        return _assistant_msg("consensus")

    agent = SwarmAgent()
    app = agent.compile(
        workers={
            "A": worker_a,
            "B": worker_b,
        },
        consensus_node=consensus,
        options={"collect": collect},
    )

    out = app.invoke(
        {"messages": [Message.from_text("start swarm")]},
        config={"thread_id": "t_swarm"},
    )
    assert isinstance(out.get("messages", []), list)  # noqa: S101
    # Ensure our nodes contributed responses
    contents = [m.content for m in out["messages"]]
    assert any("worker_a" in c for c in contents)  # noqa: S101
    assert any("worker_b" in c for c in contents)  # noqa: S101
    assert any("consensus" in c for c in contents)  # noqa: S101


def test_network_agent_static_edges():
    # Two simple nodes with a static edge between them
    def node1(state: AgentState, config: dict):
        return _assistant_msg("n1")

    def node2(state: AgentState, config: dict):
        return _assistant_msg("n2")

    agent = NetworkAgent()
    app = agent.compile(
        nodes={"A": node1, "B": node2},
        entry="A",
        static_edges=[("A", "B")],
    )

    out = app.invoke({"messages": [Message.from_text("start net")]}, config={"thread_id": "t_net"})
    assert isinstance(out.get("messages", []), list)  # noqa: S101
    contents = [m.content for m in out["messages"]]
    assert any("n1" in c for c in contents)  # noqa: S101
    assert any("n2" in c for c in contents)  # noqa: S101


def test_rag_agent_advanced_minimal():
    # Two retrievers followed by synthesize
    def r1(state: AgentState, config: dict):
        return _assistant_msg("r1")

    def r2(state: AgentState, config: dict):
        return _assistant_msg("r2")

    def synth(state: AgentState, config: dict):
        return _assistant_msg("synth")

    agent = RAGAgent()
    app = agent.compile_advanced(
        retriever_nodes=[r1, r2],
        synthesize_node=synth,
        options={},
    )

    out = app.invoke(
        {"messages": [Message.from_text("start rag")]},
        config={"thread_id": "t_rag_adv"},
    )
    assert isinstance(out.get("messages", []), list)  # noqa: S101
    contents = [m.content for m in out["messages"]]
    assert any("r1" in c for c in contents)  # noqa: S101
    assert any("r2" in c for c in contents)  # noqa: S101
    assert any("synth" in c for c in contents)  # noqa: S101
