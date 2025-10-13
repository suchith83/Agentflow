"""
Basic Plan-Act-Reflect loop using PlanActReflectAgent.

Flow (single tool round then final answer):
    1. PLAN    -> emits assistant message with a tool call (tools_calls list populated)
    2. ACT     -> ToolNode executes the requested tool (fetch_data)
    3. REFLECT -> Adds a reflection note
    4. PLAN    -> Sees tool result present, produces final answer (no further tool calls) -> END

Run:
    python examples/plan_act_reflect/basic_plan_act_reflect.py

Environment (optional):
    export OPENAI_API_KEY=...   # if you adapt this to use an LLM

This example is fully deterministic and does not require model APIs.
"""

from __future__ import annotations

from agentflow.graph.tool_node import ToolNode
from agentflow.prebuilt.agent.plan_act_reflect import PlanActReflectAgent
from agentflow.state import AgentState, Message


# ------------------------------------------------------------------------------
# Tool implementation
# ------------------------------------------------------------------------------


def fetch_data(query: str, tool_call_id: str | None = None) -> str:
    """
    Mock retrieval / computation function.

    Args:
        query: User query or transformed request
        tool_call_id: Injected tool execution id (optional, provided by framework)

    Returns:
        str: Deterministic pseudo-result
    """
    return f"[data] Queried knowledge base for: '{query[:60]}'"


tools = ToolNode([fetch_data])


# ------------------------------------------------------------------------------
# PLAN node
# ------------------------------------------------------------------------------


def plan(state: AgentState) -> AgentState:
    """
    Decide next action:
        - If no tool result yet: request fetch_data tool
        - Else: produce final answer (no tool call)
    """
    user_msgs = [m for m in state.context if m.role == "user"]
    query = user_msgs[-1].text() if user_msgs and hasattr(user_msgs[-1], "text") else "general"
    has_tool_result = any(m.role == "tool" for m in state.context)

    if not has_tool_result:
        # Emit assistant message requesting a tool execution
        msg = Message.text_message(
            f"Planning: need to gather supporting info for: '{query}'.",
            role="assistant",
        )
        # Populate tools_calls so internal heuristic routes to ACT
        msg.tools_calls = [
            {
                "id": "fetch_1",
                "name": "fetch_data",
                "arguments": {"query": query},
            }
        ]
        state.context.append(msg)
    else:
        # Final answer (no tools_calls -> heuristic will END)
        tool_msgs = [m for m in state.context if m.role == "tool"]
        payload = tool_msgs[-1].text() if tool_msgs and hasattr(tool_msgs[-1], "text") else ""
        state.context.append(
            Message.text_message(
                f"Final answer using tool output:\n{payload}",
                role="assistant",
            )
        )

    return state


# ------------------------------------------------------------------------------
# REFLECT node
# ------------------------------------------------------------------------------


def reflect(state: AgentState) -> AgentState:
    """
    Brief reflection over the tool output before returning to PLAN.
    """
    tool_msgs = [m for m in state.context if m.role == "tool"]
    if tool_msgs:
        state.context.append(
            Message.text_message(
                "Reflection: tool data received. Preparing final answer.",
                role="assistant",
            )
        )
    return state


# ------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------


def run() -> None:
    agent = PlanActReflectAgent[AgentState](state=AgentState())
    compiled = agent.compile(
        plan_node=plan,
        tool_node=tools,
        reflect_node=reflect,
    )

    initial = {
        "messages": [
            Message.text_message(
                "Summarize retrieval-augmented generation in one sentence.",
                role="user",
            )
        ]
    }

    result = compiled.invoke(initial, config={"thread_id": "plan-act-reflect-basic"})
    print("\n=== PLAN-ACT-REFLECT (Basic) Messages ===\n")
    for m in result["messages"]:
        role = getattr(m, "role", "unknown")
        print(f"[{role}] {m.text() if hasattr(m, 'text') else m}")


if __name__ == "__main__":
    run()
