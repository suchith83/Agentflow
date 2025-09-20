import logging

from pyagenity.prebuilt.agent import RAGAgent
from pyagenity.utils import Message


def plan(state, config):
    # Optionally expand/transform the query
    return Message.create(role="assistant", content="query plan ready")


def retrieve_1(state, config):
    return Message.create(role="assistant", content="r1: docs")


def retrieve_2(state, config):
    return Message.create(role="assistant", content="r2: docs")


def merge(state, config):
    return Message.create(role="assistant", content="merged")


def rerank(state, config):
    return Message.create(role="assistant", content="reranked")


def compress(state, config):
    return Message.create(role="assistant", content="compressed")


def synthesize(state, config):
    return Message.create(role="assistant", content="final answer")


def run():
    agent = RAGAgent()
    app = agent.compile_advanced(
        retriever_nodes=[retrieve_1, retrieve_2],
        synthesize_node=synthesize,
        options={
            "query_plan": plan,
            "merge": merge,
            "rerank": rerank,
            "compress": compress,
        },
    )

    res = app.invoke(
        {"messages": [Message.from_text("start rag adv")]},
        config={"thread_id": "ex-rag-adv"},
    )
    for m in res.get("messages", []):
        logging.info("[%s] %s", m.role, m.content)


if __name__ == "__main__":
    run()
