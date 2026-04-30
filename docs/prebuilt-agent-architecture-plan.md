# Prebuilt Agent Architecture Plan

Status: discussion draft
Date: 2026-04-26

This document is a planning note for which prebuilt agent patterns Agentflow should support. It is intentionally not tied to the current files in `agentflow/prebuilt/agent`; those files are useful context, but the catalog should be based on what is valuable for modern agent applications.

## 2026 Research Signals

The useful patterns in current agent systems are converging around a few ideas:

- Durable graph/workflow execution with checkpointing, streaming, interrupts, human-in-the-loop, and replayable state.
- Deep-agent harnesses that combine planning, context management, subagents, virtual files, memory, and approval gates.
- Multi-agent teams where a coordinator, supervisor, selector, or handoff mechanism routes work to specialists.
- Deterministic workflow agents for sequential, parallel, and loop execution where predictable control flow matters more than LLM-driven routing.
- Standardized tool and agent interoperability through MCP for tools/resources and A2A for cross-agent collaboration.
- Guardrails and observability as first-class production features, especially around tool calls, handoffs, inputs, outputs, traces, and approvals.

Useful references:

- LangGraph describes durable graph execution with persistence, streaming, human-in-the-loop, long-term memory, and customizable multi-agent systems: https://langchain-ai.github.io/langgraphjs/reference/modules/langgraph.html
- LangGraph checkpointing saves graph state at each superstep for human-in-the-loop, memory between interactions, and state management: https://langchain-ai.github.io/langgraphjs/reference/modules/langgraph-checkpoint.html
- LangChain Deep Agents combine planning, filesystem-backed context management, subagent delegation, long-term memory, and human-in-the-loop on top of LangGraph: https://docs.langchain.com/oss/python/deepagents/overview
- Google ADK documents common multi-agent patterns: coordinator/dispatcher, sequential pipeline, parallel fan-out/gather, hierarchical decomposition, review/critique, iterative refinement, and human-in-the-loop: https://google.github.io/adk-docs/agents/multi-agents/
- Google ADK workflow agents define deterministic sequential, parallel, and loop orchestrators: https://google.github.io/adk-docs/agents/workflow-agents/
- AutoGen AgentChat exposes multi-agent teams including round-robin, selector group chat, Magentic-One, swarm, memory, logging, and graph workflows: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html
- OpenAI Agents SDK treats handoffs, guardrails, tool calls, sessions, and tracing as core workflow concepts: https://openai.github.io/openai-agents-python/guardrails/ and https://openai.github.io/openai-agents-python/tracing/
- MCP standardizes how agents connect to external tools, prompts, and resources: https://modelcontextprotocol.info/specification/
- A2A standardizes agent-to-agent interoperability, discovery, task management, streaming, and secure information exchange across independent agent systems: https://a2aproject.github.io/A2A/specification/

## Recommended Architecture Families

Agentflow should support 8 architecture families:

1. Single-agent tool loop
2. Retrieval and knowledge workflows
3. Routing and handoff workflows
4. Deterministic workflow orchestration
5. Deep-agent long-running task harnesses
6. Multi-agent team orchestration
7. Parallel decomposition and aggregation
8. Production safety, approval, and evaluation wrappers

## Recommended Prebuilt Agents

### Tier 1: Core agents

These should be implemented first and kept stable.

1. `ReactAgent`
   - Purpose: one agent that reasons, calls tools, observes results, and answers.
   - Why useful: still the default pattern for simple and medium-complexity tool use.
   - Use skill class that will help to do anything with the tools, and the agent will learn to use them as needed.

2. `RAGAgent`
   - Purpose: retrieve context, answer with grounded information, optionally cite sources.
   - Why useful: knowledge agents remain one of the highest-value production use cases.

3. `RouterAgent`
   - Purpose: classify input and route to a node, tool, specialist, or workflow.
   - Why useful: routing is the smallest useful multi-agent primitive.

4. `SequentialAgent`
   - Purpose: run steps in strict order.
   - Why useful: deterministic business workflows need predictable execution.

5. `LoopAgent`
   - Purpose: repeat one or more steps until a condition, quality threshold, or max iteration is reached.
   - Why useful: review/fix, generate/test, critique/revise, and self-healing workflows all need bounded loops.

6. `ParallelAgent`
   - Purpose: run independent branches concurrently and collect results.
   - Why useful: lowers latency for independent retrieval, analysis, validation, or generation tasks.

### Tier 2: Advanced agents

These should build on the core agents instead of creating separate one-off abstractions.

7. `DeepAgent`
   - Purpose: long-running task harness with planning, todos, subagents, memory, context files, and checkpointed recovery.
   - Why useful: this is the most important 2026-era pattern for research, coding, data analysis, and complex operations.
   - Suggested internals: planner, executor/tool loop, virtual workspace, summarizer, subagent task tool, critic, durable checkpoint policy.

8. `DeepResearchAgent`
   - Purpose: specialized deep agent for research: plan, search/read, extract evidence, synthesize, critique, and produce report.
   - Why useful: research is a common concrete product surface and needs stronger source/evidence discipline than generic deep work.
   - Could be implemented as a configured `DeepAgent`, not necessarily a separate engine.

9. `SupervisorTeamAgent`
   - Purpose: central supervisor delegates to specialist workers and aggregates results.
   - Why useful: good for enterprise workflows where control, auditability, and explicit ownership matter.

10. `SwarmAgent`
    - Purpose: decentralized or handoff-driven specialist collaboration.
    - Why useful: good for exploratory tasks and local routing when a single supervisor would become a bottleneck.

11. `MapReduceAgent`
    - Purpose: split large input, process chunks independently, reduce into final output.
    - Why useful: useful for document sets, logs, codebase analysis, batch extraction, and large-context tasks.

12. `GuardedAgent`
    - Purpose: wrap another agent/workflow with input, output, and tool-call validation.
    - Why useful: production systems need policy, approval, and cost/safety control as reusable building blocks.

13. `HumanApprovalAgent`
    - Purpose: pause for approval, correction, or external input before continuing.
    - Why useful: high-impact tool calls, remote execution, payments, data writes, and customer-facing workflows need explicit review.

### Tier 3: Protocol and interoperability agents

These are important, but can come after the core orchestration surface is stable.

14. `MCPAgent`
    - Purpose: agent with dynamic MCP tool/resource access.
    - Why useful: MCP is becoming the standard tool integration layer.

15. `A2AAgent`
    - Purpose: expose or call remote agents through Agent2Agent-compatible tasks.
    - Why useful: agent ecosystems are moving toward cross-framework collaboration.

16. `EvaluatorAgent`
    - Purpose: score, validate, compare, or regression-test outputs from another agent.
    - Why useful: production agents need automated evaluation loops, not only runtime execution.

## Recommended Count

For a strong but manageable public prebuilt catalog:

- Support 8 architecture families.
- Implement 13 first-party prebuilt agents in the near term.
- Keep `MCPAgent`, `A2AAgent`, and `EvaluatorAgent` as protocol/evaluation extensions once the core catalog is stable.

Near-term target:

1. `ReactAgent`
2. `RAGAgent`
3. `RouterAgent`
4. `SequentialAgent`
5. `LoopAgent`
6. `ParallelAgent`
7. `DeepAgent`
8. `DeepResearchAgent`
9. `SupervisorTeamAgent`
10. `SwarmAgent`
11. `MapReduceAgent`
12. `GuardedAgent`
13. `HumanApprovalAgent`

Later extension target:

14. `MCPAgent`
15. `A2AAgent`
16. `EvaluatorAgent`

## Redis And Durable Recovery

Recommendation:

For deep agents, Redis should be treated as hot state/cache, not the source of truth. Use PostgreSQL or another durable checkpointer for authoritative recovery, and persist after every expensive/side-effectful deep-agent stage: after `PLAN`, each `RESEARCH` batch, `SYNTHESIZE`, `CRITIQUE`, and before/after interrupts. That gives crash recovery without depending on Redis survival.

Expected behavior:

- If Redis contains the latest state, resume from Redis for fast realtime continuation.
- If Redis crashes, expires, or data is removed, resume from the durable checkpointer.
- If durable state exists but Redis had newer in-flight state, resume from the last durable checkpoint and possibly repeat the last non-durable step.
- If both Redis and durable checkpoint data are gone, exact resume is impossible. The workflow must start fresh or reconstruct from external message/event logs if available.

Design implications for `DeepAgent`:

- Persist after each major stage, not only at final completion.
- Persist before and after expensive tools or side-effectful tools.
- Make side-effectful tools idempotent where possible by using operation IDs.
- Store enough stage metadata to know whether a step should retry, skip, or ask for human review after recovery.
- Keep Redis TTL configurable and document that Redis is a performance layer.
- Add recovery tests that simulate Redis loss between deep-agent stages.

## Open Questions For Discussion

- Should `DeepResearchAgent` be a separate public class, or a preset/configuration of `DeepAgent`?
- Should `LoopAgent` and `ParallelAgent` be standalone prebuilt agents, or lower-level workflow nodes used by `DeepAgent`, `MapReduceAgent`, and `SupervisorTeamAgent`?
- Should protocol-facing `MCPAgent` and `A2AAgent` live in `prebuilt/agent`, or under runtime protocol adapters?
- What should the default durable checkpoint policy be for long-running agents: every node, every stage, or only configured stage boundaries?
- Should `GuardedAgent` wrap all tool calls by default, or only input/output boundaries unless configured otherwise?
