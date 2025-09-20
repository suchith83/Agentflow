import logging

from pyagenity.prebuilt.agent import SwarmAgent
from pyagenity.utils import Message


def worker_a(state, config):
    return Message.create(role="assistant", content="worker_a: processed")


def worker_b(state, config):
    return Message.create(role="assistant", content="worker_b: processed")


def collect(state, config):
    # Optionally merge or annotate results between workers
    return Message.create(role="assistant", content="collect: merged")


def consensus(state, config):
    # Produce final consolidated output
    return Message.create(role="assistant", content="consensus: final answer")


def run():
    agent = SwarmAgent()
    app = agent.compile(
        workers={"A": worker_a, "B": worker_b},
        consensus_node=consensus,
        options={"collect": collect},
    )

    res = app.invoke(
        {"messages": [Message.from_text("start swarm")]},
        config={"thread_id": "ex-swarm"},
    )
    for m in res.get("messages", []):
        logging.info("[%s] %s", m.role, m.content)


if __name__ == "__main__":
    run()
