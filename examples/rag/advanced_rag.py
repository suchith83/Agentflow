"""
Advanced RAG (Retrieval-Augmented Generation) example using PyAgenity's RAGAgent.compile_advanced.

Chain (conceptual):
    QUERY_PLAN -> RETRIEVE_1 (dense) -> MERGE -> RETRIEVE_2 (sparse) -> MERGE
        -> RERANK -> COMPRESS -> SYNTHESIZE -> (END)

Each stage here is a lightweight placeholder to illustrate how to compose a
hybrid retrieval pipeline. Replace internals with actual vector / keyword /
BM25 / rerank / summarization logic in production.

Run:
    python examples/rag/advanced_rag.py

Env (LiteLLM compatible):
    export OPENAI_API_KEY=...          # or another provider key
    export RAG_MODEL=gpt-4o-mini       # optional override

Key Ideas:
- Multiple retrievers (dense + sparse) feeding a merge step
- Optional pipeline nodes (query planning, rerank, compress)
- Clean separation of responsibilities
- Async-friendly nodes (all defined async for realism)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, List


try:
    from litellm import acompletion
except Exception:  # pragma: no cover
    acompletion = None  # type: ignore

from pyagenity.prebuilt.agent.rag import RAGAgent
from pyagenity.state import AgentState, Message
from pyagenity.utils.constants import END


# -----------------------------------------------------------------------------
# Mock Corpora / Indices (replace with real retrieval backends)
# -----------------------------------------------------------------------------
DENSE_CORPUS = [
    {"id": "d1", "text": "PyAgenity enables composable agent graphs."},
    {"id": "d2", "text": "RAG pipelines combine retrieval with LLM synthesis."},
    {"id": "d3", "text": "Hybrid retrieval blends dense and sparse search signals."},
]

SPARSE_CORPUS = [
    {"id": "s1", "text": "Sparse retrieval uses lexical term matching (BM25 / TF-IDF)."},
    {"id": "s2", "text": "Vector stores like Qdrant support semantic similarity search."},
    {"id": "s3", "text": "Merging strategies must reconcile duplicates and scoring scales."},
]


# -----------------------------------------------------------------------------
# Utility helpers (placeholder scoring)
# -----------------------------------------------------------------------------
def keyword_score(query: str, text: str) -> int:
    q_tokens = set(query.lower().split())
    return sum(1 for t in q_tokens if t in text.lower())


def extract_latest_user_text(state: AgentState) -> str:
    for msg in reversed(state.context):
        if msg.role == "user" and hasattr(msg, "text"):
            return msg.text()  # type: ignore
    return ""


def append_context(state: AgentState, label: str, content: str) -> None:
    state.context.append(Message.text_message(f"[{label}]\n{content}", role="assistant"))


# -----------------------------------------------------------------------------
# Pipeline Nodes
# -----------------------------------------------------------------------------
async def query_plan(state: AgentState) -> AgentState:
    """Refine the user query (placeholder)."""
    base_query = extract_latest_user_text(state)
    if not base_query:
        return state
    # Simple heuristic refinement
    refined = base_query.strip().rstrip("?")
    plan_note = f"Refined query: {refined}"
    append_context(state, "query_plan", plan_note)
    return state


async def dense_retriever(state: AgentState) -> AgentState:
    """Dense retriever placeholder (scores by naive keyword overlap)."""
    query = extract_latest_user_text(state)
    if not query:
        return state
    scored: List[tuple[int, dict[str, str]]] = [
        (keyword_score(query, doc["text"]), doc) for doc in DENSE_CORPUS
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d["text"] for score, d in scored if score > 0][:2]
    content = "\n".join(f"- {t}" for t in top) if top else "No dense matches."
    append_context(state, "dense_retrieval", content)
    return state


async def sparse_retriever(state: AgentState) -> AgentState:
    """Sparse retriever placeholder (same heuristic)."""
    query = extract_latest_user_text(state)
    if not query:
        return state
    scored: List[tuple[int, dict[str, str]]] = [
        (keyword_score(query, doc["text"]), doc) for doc in SPARSE_CORPUS
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d["text"] for score, d in scored if score > 0][:2]
    content = "\n".join(f"- {t}" for t in top) if top else "No sparse matches."
    append_context(state, "sparse_retrieval", content)
    return state


async def merge_stage(state: AgentState) -> AgentState:
    """Merge retrieval outputs (deduplicate naive)."""
    dense_msgs = [
        m for m in state.context if "[dense_retrieval]" in (m.text() if hasattr(m, "text") else "")
    ]
    sparse_msgs = [
        m for m in state.context if "[sparse_retrieval]" in (m.text() if hasattr(m, "text") else "")
    ]
    combined = []
    for msg in dense_msgs + sparse_msgs:
        txt = msg.text() if hasattr(msg, "text") else ""
        for line in txt.splitlines():
            if line.startswith("- "):
                norm = line.strip()
                if norm not in combined:
                    combined.append(norm)
    if not combined:
        merged = "No merged documents."
    else:
        merged = "\n".join(combined)
    append_context(state, "merge", merged)
    return state


async def rerank_stage(state: AgentState) -> AgentState:
    """Rerank merged docs (placeholder keeps order)."""
    merged_msgs = [
        m for m in state.context if "[merge]" in (m.text() if hasattr(m, "text") else "")
    ]
    if not merged_msgs:
        return state
    # Pass-through; in real usage apply ML reranker or score normalization
    append_context(state, "rerank", "Rerank applied (no-op placeholder).")
    return state


async def compress_stage(state: AgentState) -> AgentState:
    """Compress documents (placeholder summarizer)."""
    merged_msgs = [
        m for m in state.context if "[merge]" in (m.text() if hasattr(m, "text") else "")
    ]
    if not merged_msgs:
        return state
    merged_text = merged_msgs[-1].text() if hasattr(merged_msgs[-1], "text") else ""
    lines = [l[2:].strip() for l in merged_text.splitlines() if l.startswith("- ")]
    if not lines:
        summary = "No content to compress."
    else:
        # Simple truncation summarizer
        summary = " + ".join(lines[:2])
        if len(lines) > 2:
            summary += " ..."
    append_context(state, "compress", summary)
    return state


async def synthesize(state: AgentState) -> AgentState:
    """Generate final answer using the retrieved / compressed context."""
    query = extract_latest_user_text(state)
    compress_msgs = [
        m for m in state.context if "[compress]" in (m.text() if hasattr(m, "text") else "")
    ]
    merged_msgs = [
        m for m in state.context if "[merge]" in (m.text() if hasattr(m, "text") else "")
    ]
    context_block = ""
    if compress_msgs:
        context_block = compress_msgs[-1].text() if hasattr(compress_msgs[-1], "text") else ""
    elif merged_msgs:
        context_block = merged_msgs[-1].text() if hasattr(merged_msgs[-1], "text") else ""

    model = os.getenv("RAG_MODEL", "gpt-4o-mini")

    if acompletion:
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a concise assistant answering using provided retrieval context.",
                },
                {
                    "role": "user",
                    "content": f"Question:\n{query}\n\nContext:\n{context_block}",
                },
            ]
            resp = await acompletion(model=model, messages=messages)
            # Guard against None / unexpected types for static type checkers
            raw_answer = getattr(resp.choices[0].message, "content", None)  # type: ignore[attr-defined]
            answer = raw_answer if isinstance(raw_answer, str) and raw_answer else ""
        except Exception as e:  # pragma: no cover
            answer = f"(Fallback) Could not call model: {e}\nContext:\n{context_block}"
    else:
        answer = f"(Local synthesis) Q: {query}\nContext summary: {context_block[:180]}"

    safe_answer = answer or "(No answer generated)"
    state.context.append(Message.text_message(safe_answer, role="assistant"))
    return state


# -----------------------------------------------------------------------------
# Follow-up condition (always END for this demo)
# -----------------------------------------------------------------------------
def end_condition(_: AgentState) -> str:
    return END


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
async def main() -> None:
    rag = RAGAgent[AgentState](state=AgentState())

    compiled = rag.compile_advanced(
        retriever_nodes=[dense_retriever, sparse_retriever],
        synthesize_node=synthesize,
        options={
            "query_plan": query_plan,
            "merge": merge_stage,
            "rerank": rerank_stage,
            "compress": compress_stage,
            "followup_condition": end_condition,
        },
    )

    initial = {
        "messages": [
            Message.text_message(
                "Explain how a hybrid RAG pipeline works and why reranking helps.",
                role="user",
            )
        ]
    }

    result = await compiled.ainvoke(initial, config={"thread_id": "advanced-rag"})
    print("\n=== ADVANCED RAG PIPELINE RESULT ===\n")
    for msg in result["messages"]:
        role = getattr(msg, "role", "unknown")
        print(f"[{role}] {msg}")


if __name__ == "__main__":
    asyncio.run(main())
