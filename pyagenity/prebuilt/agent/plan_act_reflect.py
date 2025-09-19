from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from injectq import InjectQ

from pyagenity.checkpointer.base_checkpointer import BaseCheckpointer
from pyagenity.graph.compiled_graph import CompiledGraph
from pyagenity.graph.state_graph import StateGraph
from pyagenity.graph.tool_node import ToolNode
from pyagenity.publisher.base_publisher import BasePublisher
from pyagenity.state.agent_state import AgentState
from pyagenity.state.base_context import BaseContextManager
from pyagenity.store.base_store import BaseStore
from pyagenity.utils.callbacks import CallbackManager
from pyagenity.utils.constants import END
from pyagenity.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


StateT = TypeVar("StateT", bound=AgentState)


def _should_act(state: AgentState) -> str:
    """Decide whether to perform tool-use (ACT) or finish.

    Heuristic:
    - If last assistant message proposes tool calls -> go to ACT
    - If last message is a tool result -> go to REFLECT
    - Otherwise END
    """
    if not state.context:
        return END

    last = state.context[-1]
    # If assistant asked for tools
    if (
        getattr(last, "tools_calls", None)
        and isinstance(last.tools_calls, list)
        and len(last.tools_calls) > 0
        and last.role == "assistant"
    ):
        return "ACT"

    # If tool responded, reflect
    if last.role == "tool" and last.tool_call_id is not None:
        return "REFLECT"

    return END


class PlanActReflectAgent[StateT: AgentState]:
    """A plan -> act(tool) -> reflect loop agent.

    Nodes:
    - PLAN: produce a plan or next action; may request tools
    - ACT: ToolNode executes requested tools
    - REFLECT: analyze tool results and decide to loop back to PLAN or END

    Edges:
    PLAN -> conditional(_should_act): {"ACT": ACT, "REFLECT": REFLECT, END: END}
    ACT -> REFLECT
    REFLECT -> PLAN
    Entry: PLAN
    """

    def __init__(
        self,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: InjectQ | None = None,
    ):
        self._graph = StateGraph[StateT](
            state=state,
            context_manager=context_manager,
            publisher=publisher,
            id_generator=id_generator,
            container=container,
        )

    def compile(
        self,
        plan_node: Callable,
        tool_node: ToolNode,
        reflect_node: Callable,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        if not callable(plan_node):
            raise ValueError("plan_node must be callable")
        if not isinstance(tool_node, ToolNode):
            raise ValueError("tool_node must be a ToolNode")
        if not callable(reflect_node):
            raise ValueError("reflect_node must be callable")

        self._graph.add_node("PLAN", plan_node)
        self._graph.add_node("ACT", tool_node)
        self._graph.add_node("REFLECT", reflect_node)

        # PLAN decides next step
        self._graph.add_conditional_edges(
            "PLAN",
            _should_act,
            {"ACT": "ACT", "REFLECT": "REFLECT", END: END},
        )

        # Loop
        self._graph.add_edge("ACT", "REFLECT")
        self._graph.add_edge("REFLECT", "PLAN")

        # Entry
        self._graph.set_entry_point("PLAN")

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
