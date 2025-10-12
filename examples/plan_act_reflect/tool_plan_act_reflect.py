"""
Tool-centric Plan-Act-Reflect example demonstrating:
  * Multiple tool functions registered in a ToolNode
  * Custom decision function (condition=...) overriding default heuristic
  * Iterative loop until explicit stop criteria (max iterations or confidence)

Loop Logic:
    PLAN -> (condition) -> ACT | REFLECT | END
    ACT  -> REFLECT
    REFLECT -> PLAN

Stop Criteria:
    - Max iterations reached
    - PLAN emits final answer (no tool calls & confidence high)

Run:
    python examples/plan_act_reflect/tool_plan_act_reflect.py
"""

import re
from dataclasses import dataclass

from pyagenity.graph.tool_node import ToolNode
from pyagenity.prebuilt.agent.plan_act_reflect import PlanActReflectAgent
from pyagenity.state import AgentState, Message
from pyagenity.utils.constants import END


# ------------------------------------------------------------------------------
# Simple in-memory "state extensions"
# ------------------------------------------------------------------------------


@dataclass
class LoopMeta:
    turns: int = 0
    last_confidence: float = 0.0


def _ensure_meta(state: AgentState) -> LoopMeta:
    meta = getattr(state, "_loop_meta", None)
    if not isinstance(meta, LoopMeta):
        meta = LoopMeta()
        setattr(state, "_loop_meta", meta)
    return meta


# ------------------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------------------


def search_corpus(query: str) -> str:
    """
    Naive keyword search over a tiny corpus. Replace with vector / API search.
    """
    corpus = {
        "rag": "RAG combines retrieval with generation for grounded responses.",
        "graph": "StateGraph wires nodes with conditional edges for agent control flow.",
        "tools": "ToolNode executes registered functions or external adapters.",
    }
    hits = [v for k, v in corpus.items() if k in query.lower()]
    return " | ".join(hits) if hits else "No matches."


def extract_numbers(text: str) -> str:
    """
    Extract integers & compute sum (toy analytical tool).
    """
    nums = list(map(int, re.findall(r"\b\d+\b", text)))
    if not nums:
        return "No numbers found."
    return f"Numbers: {nums}; sum={sum(nums)}"


tools = ToolNode([search_corpus, extract_numbers])


# ------------------------------------------------------------------------------
# PLAN
# ------------------------------------------------------------------------------


def plan(state: AgentState) -> AgentState:
    """
    Decide which tool(s) to call based on latest user / assistant context.
    Emits either:
        - Assistant message with tool calls
        - Assistant final answer (no tool calls)
    """
    meta = _ensure_meta(state)
    meta.turns += 1

    user_msgs = [m for m in state.context if m.role == "user"]
    query = user_msgs[-1].text() if user_msgs and hasattr(user_msgs[-1], "text") else ""

    # If we already produced a reflected tool result and confidence OK -> finalize
    tool_msgs = [m for m in state.context if m.role == "tool"]
    if tool_msgs and meta.last_confidence >= 0.8:
        state.context.append(
            Message.text_message(
                f"Final answer (confidence={meta.last_confidence:.2f}):\n{tool_msgs[-1].text()}",
                role="assistant",
            )
        )
        return state

    desired_tools = []
    if any(k in query.lower() for k in ["rag", "graph", "tool"]):
        desired_tools.append(
            {
                "id": f"search_{meta.turns}",
                "name": "search_corpus",
                "arguments": {"query": query},
            }
        )

    if re.search(r"\d+", query):
        desired_tools.append(
            {
                "id": f"extract_{meta.turns}",
                "name": "extract_numbers",
                "arguments": {"text": query},
            }
        )

    if not desired_tools:
        # Nothing to call -> produce a generic expansion then END in condition
        state.context.append(
            Message.text_message("No tools needed; answering directly.", role="assistant")
        )
        meta.last_confidence = 0.9
        return state

    msg = Message.text_message(
        f"Planning turn {meta.turns}: invoking {', '.join(t['name'] for t in desired_tools)}",
        role="assistant",
    )
    msg.tools_calls = desired_tools
    state.context.append(msg)
    return state


# ------------------------------------------------------------------------------
# REFLECT
# ------------------------------------------------------------------------------


def reflect(state: AgentState) -> AgentState:
    """
    Adjust confidence based on tool outputs (rudimentary scoring).
    """
    meta = _ensure_meta(state)
    tool_msgs = [m for m in state.context if m.role == "tool"]
    if tool_msgs:
        latest = tool_msgs[-1].text() if hasattr(tool_msgs[-1], "text") else ""
        # Naive confidence heuristic: length-based
        meta.last_confidence = min(1.0, max(0.1, len(latest) / 120))
        state.context.append(
            Message.text_message(
                f"Reflection: updated confidence={meta.last_confidence:.2f}",
                role="assistant",
            )
        )
    return state


# ------------------------------------------------------------------------------
# Custom decision function
# ------------------------------------------------------------------------------


def routing_condition(state: AgentState) -> str:
    """
    Decide next path:
        - If last assistant message has tools_calls -> ACT
        - If last message is tool -> REFLECT
        - If PLAN produced final answer or max turns -> END
        - Else -> REFLECT (forces one reflection before next PLAN)
    """
    meta = _ensure_meta(state)
    if meta.turns >= 6:
        return END
    if not state.context:
        return END

    last = state.context[-1]

    if (
        last.role == "assistant"
        and getattr(last, "tools_calls", None)
        and isinstance(last.tools_calls, list)
        and last.tools_calls
    ):
        return "ACT"

    if last.role == "tool":
        return "REFLECT"

    # If final answer message present (confidence high)
    if last.role == "assistant" and "Final answer" in (
        last.text() if hasattr(last, "text") else ""
    ):
        return END

    return "REFLECT"


# ------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------


def run() -> None:
    agent = PlanActReflectAgent[AgentState](state=AgentState())
    compiled = agent.compile(
        plan_node=plan,
        tool_node=tools,
        reflect_node=reflect,
        condition=routing_condition,  # override default heuristic
    )

    initial = {
        "messages": [
            Message.text_message(
                "Please search rag and graph concepts and also add 4 9 13 2.",
                role="user",
            )
        ]
    }

    result = compiled.invoke(initial, config={"thread_id": "plan-act-reflect-tools"})
    print("\n=== PLAN-ACT-REFLECT (Multi-Tool) Messages ===\n")
    for m in result["messages"]:
        role = getattr(m, "role", "unknown")
        print(f"[{role}] {m.text() if hasattr(m, 'text') else m}")


if __name__ == "__main__":
    run()
