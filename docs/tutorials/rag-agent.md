---
title: RAG Agent
---

# RAG Agent

Two ways to build Retrieval-Augmented Generation (RAG) pipelines:

1) Simple RAG: Retrieve -> Synthesize with optional follow-up

```python
from pyagenity.prebuilt.agent import RAGAgent
from pyagenity.utils import Message

def retrieve(state, config):
    return Message.create(role="assistant", content="docs")

def synthesize(state, config):
    return Message.create(role="assistant", content="answer")

agent = RAGAgent()
app = agent.compile(retriever_node=retrieve, synthesize_node=synthesize)
out = app.invoke({"messages": [Message.from_text("question")]}, config={"thread_id": "rag"})
```

2) Advanced RAG: hybrid retrieval with optional stages

```python
agent = RAGAgent()
app = agent.compile_advanced(
    retriever_nodes=[retrieve1, retrieve2],
    synthesize_node=synthesize,
    options={
        "query_plan": plan,   # optional
        "merge": merge,       # optional
        "rerank": rerank,     # optional
        "compress": compress, # optional
    },
)
```

Best practices to consider:
- Hybrid retrieval (sparse + dense)
- Multi-query or HyDE expansion
- MMR or learning-to-rerank
- Context compression to fit model limits
