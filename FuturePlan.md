Tier 1 — high leverage, fills real gaps
Token/cost accounting middleware. A UsageTracker that aggregates input/output/cache tokens + USD per provider, per node, per thread. Surface it on AgentState.usage and as a stream event. Everyone builds this themselves badly; ship it once, correctly.
Retry + circuit breaker for LLM calls. Exponential backoff with jitter, separate policies for rate-limit (429), server (5xx), and tool errors. Configure on Agent(retry=RetryPolicy(...)). Today users wrap your SDK in tenacity — make it a first-class concern.
Structured output / response schemas. First-class response_format=PydanticModel across providers, with automatic re-prompt on validation failure. OpenAI has it, Google has it, you should expose one API that hides the difference.
Prompt caching as a primitive. Anthropic + OpenAI both expose cache controls. Add cache_segments=["system", "tools"] on Agent and let the provider adapter translate. This is the single biggest cost lever for real apps; making it declarative is a moat.
Human-in-the-loop / interrupt primitives. You have interrupt_before/after hooks in CompiledGraph — turn them into a real interrupt() callable inside a node that suspends, persists via checkpointer, and resumes on compiled.resume(thread_id, value). This is what LangGraph's interrupt does and it's table stakes for approval workflows.
OpenTelemetry tracing out of the box. You already emit events via Publisher. Add an OTelPublisher that maps to OTel spans (one span per node, child spans per tool call, attributes for tokens/model/cost). Production users need this on day one.
Tier 2 — convenience that competitors charge for
Eval harness with criteria runners. You have a qa/ folder — push it further: dataset loader, parallel runner, LLM-as-judge primitives, regression diffing against a baseline run. The recent eval commits suggest you're heading there; commit to it.
Built-in guardrails node. Input/output classifiers, PII redaction, jailbreak detection, max-token/cost ceilings per thread. Compose as a GuardrailNode that can sit between START and the first agent.
Semantic + summary memory. Working state is fine. Add automatic conversation summarization when context approaches the model limit, and a vector-recall memory that injects top-k relevant turns. You already have Qdrant/Mem0 extras — wire them into a default MemoryManager so users don't roll their own.
More prebuilt agents that actually ship. Finish the commented-out ones (Sequential, Supervisor, Swarm, MapReduce, PlanActReflect) or delete them. If I had to pick three to land: Supervisor, PlanActReflect, MapReduce — those cover 80% of real multi-agent patterns.
Streaming structured deltas. Today users get content chunks. Add typed deltas: ToolCallDelta, ReasoningDelta, UsageDelta, StateDelta. Lets the TS client render rich UI without reverse-engineering chunk shapes.
Tier 3 — depth / polish
Provider-native batch API support. Anthropic and OpenAI batch endpoints are 50% cheaper. Add compiled.batch_invoke([inputs]) that uses them when the provider supports it.
Subgraph / nested graph composition. Let a node be a CompiledGraph itself. Critical for reuse — today users copy-paste subflows.
Async-native checkpointer for SQLite/DuckDB. Postgres is great for prod but local dev needs zero-config persistence. A built-in SQLite checkpointer makes the getting-started story 10x smoother.
@tool decorator with auto-schema from docstring + types. If you already have this, expose docstring-style param descriptions (Google/NumPy style) so users don't repeat themselves. Pydantic-based.
Deterministic replay mode. Given a thread_id, re-run the graph using recorded LLM responses from the checkpointer. Game-changer for debugging and tests.
Rate limiter as a graph primitive, not just API-layer. Per-tool, per-user, per-model. The API package has it; the core should too.
Tier 4 — would be nice, lower priority
A2A protocol support (already in extras — finish it).
Plugin entry-point system (agentflow.tools group) so third parties can publish agentflow-tools-stripe etc.
CLI codegen: agentflow generate agent --from-openapi spec.yaml to scaffold tools from an OpenAPI spec.
Cost-aware routing: a RouterAgent mode that picks cheaper models for simple queries based on a classifier.
What I would NOT add
More provider adapters until OpenAI/Anthropic/Google are rock solid. Litellm exists for the long tail.
A built-in vector DB. You have Qdrant; that's enough. Don't grow this into LangChain.
Yet another agent pattern abstraction. Three solid prebuilts beat ten half-built ones (see review).
GUI/visualization in the Python core. Keep that in the playground.
If I could only ship 3 in the next release
Token/cost tracking + prompt caching declarative API (Tier 1 #1, #4) — biggest user pain.
OTel publisher (Tier 1 #6) — unlocks every observability story.
Interrupt/resume + SQLite checkpointer (Tier 1 #5 + Tier 3 #14) — turns demos into apps.
Those three move you from "yet another agent framework" to "the one you'd actually bet a product on." Everything else can wait for 0.9/1.0.