"""
Basic RAG (Retrieval-Augmented Generation) example using TAF's RAGAgent.

Pattern:
1. RETRIEVE  - Collect relevant context for the latest user query
2. SYNTHESIZE - Produce an answer using retrieved context
3. (Optional follow-up loop) - Omitted here (ends after one synth)

Run:
    python examples/rag/basic_rag.py

Environment:
    export OPENAI_API_KEY=...        # or provider key supported by LiteLLM
    export RAG_MODEL=gpt-4o-mini     # (optional) any LiteLLM-compatible model

Key Points:
- Demonstrates minimal usage of RAGAgent
- In-memory pseudo-retriever with trivial scoring
- Synthesis step uses LiteLLM if available, else falls back to string response
"""

import os
from typing import List


try:
    # Prefer async interface but we keep retrieval synchronous for simplicity
    from litellm import completion
except Exception:  # pragma: no cover - optional dependency
    completion = None  # type: ignore

from agentflow.prebuilt.agent.rag import RAGAgent
from agentflow.state import AgentState, Message
from agentflow.utils.constants import END


# -----------------------------------------------------------------------------
# Mock Knowledge Base (Replace with vector store / embedding retriever in prod)
# -----------------------------------------------------------------------------
DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "doc1",
        "text": "Retrieval-Augmented Generation (RAG) combines document retrieval with LLM synthesis.",
    },
    {
        "id": "doc2",
        "text": "TAF StateGraph lets you wire custom nodes and conditional edges.",
    },
    {
        "id": "doc3",
        "text": "Hybrid retrieval can mix sparse, dense, and metadata filtering strategies.",
    },
    {
        "id": "doc4",
        "text": "Use a vector store (e.g., Qdrant) for semantic similarity in production RAG systems.",
    },
]


# -----------------------------------------------------------------------------
# Retriever Node
# -----------------------------------------------------------------------------
def simple_retriever(state: AgentState) -> AgentState:
    """
    Naive retrieval:
    - Extract latest user message
    - Score docs by keyword overlap
    - Keep top-k (k=2)
    - Append a tool-style message with retrieved context
    """
    if not state.context:
        return state

    # Find latest user content
    user_msgs = [m for m in state.context if m.role == "user"]
    if not user_msgs:
        return state
    query = user_msgs[-1].text() if hasattr(user_msgs[-1], "text") else ""

    if not query:
        return state

    tokens = set(query.lower().split())
    scored: List[tuple[int, dict[str, str]]] = []
    for doc in DOCUMENTS:
        overlap = sum(1 for t in tokens if t in doc["text"].lower())
        scored.append((overlap, doc))

    # Select top 2 with any overlap
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [d["text"] for score, d in scored if score > 0][:2]

    if not top_docs:
        retrieved = "No relevant documents found."
    else:
        retrieved = "\n".join(f"- {t}" for t in top_docs)

    retrieval_message = Message.text_message(
        f"[retrieval]\n{retrieved}",
        role="assistant",
    )
    state.context.append(retrieval_message)
    return state


# -----------------------------------------------------------------------------
# Synthesize Node
# -----------------------------------------------------------------------------
def synthesize_answer(state: AgentState) -> AgentState:
    """
    Synthesize an answer referencing retrieved context.

    Uses LiteLLM if available; otherwise falls back to a simple rule-based answer.
    """
    model = os.getenv("RAG_MODEL", "gpt-4o-mini")
    latest_user = next((m for m in reversed(state.context) if m.role == "user"), None)
    retrieval_ctx = [
        m for m in state.context if "[retrieval]" in (m.text() if hasattr(m, "text") else "")
    ]

    user_question = latest_user.text() if latest_user and hasattr(latest_user, "text") else ""
    retrieval_text = retrieval_ctx[-1].text() if retrieval_ctx else ""

    if completion:
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You answer user questions using provided retrieved context.",
                },
                {
                    "role": "user",
                    "content": f"Question:\n{user_question}\n\nContext:\n{retrieval_text}",
                },
            ]
            resp = completion(model=model, messages=messages)
            answer = resp.choices[0].message["content"]  # type: ignore[index]
        except Exception as e:  # pragma: no cover - fallback path
            answer = (
                f"Fallback synthesis (error invoking model: {e}).\nContext used:\n{retrieval_text}"
            )
    else:  # No LiteLLM available
        answer = f"(Local fallback) Answer to: '{user_question}'. Context:\n{retrieval_text}"

    state.context.append(Message.text_message(answer, role="assistant"))
    return state


# -----------------------------------------------------------------------------
# Optional follow-up condition (here we terminate immediately)
# -----------------------------------------------------------------------------
def end_after_first(_: AgentState) -> str:
    return END


# -----------------------------------------------------------------------------
# Build & Run
# -----------------------------------------------------------------------------
def run_sync_demo() -> None:
    """Compile and invoke the basic RAG flow."""
    rag = RAGAgent[AgentState](state=AgentState())

    compiled = rag.compile(
        retriever_node=simple_retriever,
        synthesize_node=synthesize_answer,
        followup_condition=end_after_first,  # ensures single pass
    )

    initial_input = {
        "messages": [
            Message.text_message("Explain how RAG systems work with TAF.", role="user")
        ]
    }

    result = compiled.invoke(initial_input, config={"thread_id": "basic-rag-demo"})
    print("\n=== RAG RESULT MESSAGES ===\n")
    for msg in result["messages"]:
        role = getattr(msg, "role", "unknown")
        print(f"[{role}] {msg}")


if __name__ == "__main__":
    run_sync_demo()
