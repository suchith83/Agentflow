import logging

from pyagenity.prebuilt.agent import NetworkAgent
from pyagenity.utils import Message


def n1(state, config):
    return Message.create(role="assistant", content="n1")


def n2(state, config):
    return Message.create(role="assistant", content="n2")


def run():
    agent = NetworkAgent()
    app = agent.compile(
        nodes={"A": n1, "B": n2},
        entry="A",
        static_edges=[("A", "B")],
    )

    res = app.invoke(
        {"messages": [Message.from_text("start net")]},
        config={"thread_id": "ex-net"},
    )
    for m in res.get("messages", []):
        logging.info("[%s] %s", m.role, m.content)


if __name__ == "__main__":
    run()
