from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from injectq import InjectQ

from agentflow.checkpointer.base_checkpointer import BaseCheckpointer
from agentflow.graph.compiled_graph import CompiledGraph
from agentflow.graph.state_graph import StateGraph
from agentflow.graph.tool_node import ToolNode
from agentflow.publisher.base_publisher import BasePublisher
from agentflow.state.agent_state import AgentState
from agentflow.state.base_context import BaseContextManager
from agentflow.state.execution_state import ExecutionState
from agentflow.store.base_store import BaseStore
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.constants import END
from agentflow.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


StateT = TypeVar("StateT", bound=AgentState)


def _route_after_plan(state: AgentState) -> str:
    """Router after PLAN.

    Heuristics:
    - If last assistant message proposes tool calls => RESEARCH
    - Else if context exists => SYNTHESIZE
    - Else END
    """
    if not state.context:
        return END

    last = state.context[-1]
    if (
        getattr(last, "tools_calls", None)
        and isinstance(last.tools_calls, list)
        and len(last.tools_calls) > 0
        and last.role == "assistant"
    ):
        return "RESEARCH"

    return "SYNTHESIZE"


def _route_after_critique(state: AgentState) -> str:
    """Router after CRITIQUE to decide iterate or finish.

    Uses execution_meta.internal_data to track iteration counters.
    Expects keys: dr_iters (int), dr_max_iters (int), dr_heavy_mode (bool)
    """
    em = state.execution_meta
    internal = em.internal_data

    # Defaults
    iters = int(internal.get("dr_iters", 0))
    max_iters = int(internal.get("dr_max_iters", 2))
    heavy_mode = bool(internal.get("dr_heavy_mode", False))

    # If heavy_mode, allow one extra loop through RESEARCH when critique suggests gaps
    # Heuristic: if last assistant message has tool calls, loop; otherwise finish.
    # Respect max_iters regardless of heavy_mode.
    if iters >= max_iters:
        return END

    # Check if we need more research
    if state.context:
        last = state.context[-1]
        needs_more_research = (
            getattr(last, "tools_calls", None)
            and isinstance(last.tools_calls, list)
            and len(last.tools_calls) > 0
            and last.role == "assistant"
        )
    else:
        needs_more_research = False

    if needs_more_research or heavy_mode:
        # increment iteration counter to avoid infinite loops
        em.internal_data["dr_iters"] = iters + 1
        return "RESEARCH"

    return END


class DeepResearchAgent[StateT: AgentState]:
    """Deep Research Agent: PLAN → RESEARCH → SYNTHESIZE → CRITIQUE loop.

    This agent mirrors modern deep-research patterns inspired by DeerFlow and
    Tongyi DeepResearch: plan tasks, use tools to research, synthesize findings,
    critique gaps and iterate a bounded number of times.

    Nodes:
    - PLAN: Decompose problem, propose search/tool actions; may include tool calls
    - RESEARCH: ToolNode executes search/browse/calc/etc tools
    - SYNTHESIZE: Aggregate and draft a coherent report or partial answer
    - CRITIQUE: Identify gaps, contradictions, or follow-ups; can request more tools

        Routing:
        - PLAN -> conditional(_route_after_plan):
            {"RESEARCH": RESEARCH, "SYNTHESIZE": SYNTHESIZE, END: END}
    - RESEARCH -> SYNTHESIZE
    - SYNTHESIZE -> CRITIQUE
    - CRITIQUE -> conditional(_route_after_critique): {"RESEARCH": RESEARCH, END: END}

    Iteration Control:
    - Uses execution_meta.internal_data keys:
        dr_max_iters (int): maximum critique→research loops (default 2)
        dr_iters (int): current loop count (auto-updated)
        dr_heavy_mode (bool): if True, bias towards one more loop when critique suggests
    """

    def __init__(
        self,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: InjectQ | None = None,
        max_iters: int = 2,
        heavy_mode: bool = False,
    ):
        # initialize graph
        self._graph = StateGraph[StateT](
            state=state,
            context_manager=context_manager,
            publisher=publisher,
            id_generator=id_generator,
            container=container,
        )
        # seed default internal config on prototype state
        # Note: These values will be copied to new state at invoke time.
        exec_meta: ExecutionState = self._graph._state.execution_meta
        exec_meta.internal_data.setdefault("dr_max_iters", max(0, int(max_iters)))
        exec_meta.internal_data.setdefault("dr_iters", 0)
        exec_meta.internal_data.setdefault("dr_heavy_mode", bool(heavy_mode))

    def compile(  # noqa: PLR0912
        self,
        plan_node: Callable | tuple[Callable, str],
        research_tool_node: ToolNode | tuple[ToolNode, str],
        synthesize_node: Callable | tuple[Callable, str],
        critique_node: Callable | tuple[Callable, str],
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        # Handle plan_node
        if isinstance(plan_node, tuple):
            plan_func, plan_name = plan_node
            if not callable(plan_func):
                raise ValueError("plan_node[0] must be callable")
        else:
            plan_func = plan_node
            plan_name = "PLAN"
            if not callable(plan_func):
                raise ValueError("plan_node must be callable")

        # Handle research_tool_node
        if isinstance(research_tool_node, tuple):
            research_func, research_name = research_tool_node
            if not isinstance(research_func, ToolNode):
                raise ValueError("research_tool_node[0] must be a ToolNode")
        else:
            research_func = research_tool_node
            research_name = "RESEARCH"
            if not isinstance(research_func, ToolNode):
                raise ValueError("research_tool_node must be a ToolNode")

        # Handle synthesize_node
        if isinstance(synthesize_node, tuple):
            synthesize_func, synthesize_name = synthesize_node
            if not callable(synthesize_func):
                raise ValueError("synthesize_node[0] must be callable")
        else:
            synthesize_func = synthesize_node
            synthesize_name = "SYNTHESIZE"
            if not callable(synthesize_func):
                raise ValueError("synthesize_node must be callable")

        # Handle critique_node
        if isinstance(critique_node, tuple):
            critique_func, critique_name = critique_node
            if not callable(critique_func):
                raise ValueError("critique_node[0] must be callable")
        else:
            critique_func = critique_node
            critique_name = "CRITIQUE"
            if not callable(critique_func):
                raise ValueError("critique_node must be callable")

        # Add nodes
        self._graph.add_node(plan_name, plan_func)
        self._graph.add_node(research_name, research_func)
        self._graph.add_node(synthesize_name, synthesize_func)
        self._graph.add_node(critique_name, critique_func)

        # Edges
        self._graph.add_conditional_edges(
            plan_name,
            _route_after_plan,
            {research_name: research_name, synthesize_name: synthesize_name, END: END},
        )
        self._graph.add_edge(research_name, synthesize_name)
        self._graph.add_edge(synthesize_name, critique_name)
        self._graph.add_conditional_edges(
            critique_name,
            _route_after_critique,
            {research_name: research_name, END: END},
        )

        # Entry
        self._graph.set_entry_point(plan_name)

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
