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
from agentflow.store.base_store import BaseStore
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.constants import END
from agentflow.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


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
    if last.role == "tool" and last is not None:
        return "REFLECT"

    return END


class PlanActReflectAgent[StateT: AgentState]:
    """Plan -> Act -> Reflect looping agent.

    Pattern:
        PLAN -> (condition) -> ACT | REFLECT | END
        ACT -> REFLECT
        REFLECT -> PLAN

    Default condition (_should_act):
        - If last assistant message contains tool calls -> ACT
        - If last message is from a tool -> REFLECT
        - Else -> END

    Provide a custom condition to override this heuristic and implement:
        * Budget / depth limiting
        * Confidence-based early stop
        * Dynamic branch selection (e.g., different tool nodes)

    Parameters (constructor):
        state: Optional initial state instance
        context_manager: Custom context manager
        publisher: Optional publisher for streaming / events
        id_generator: ID generation strategy
        container: InjectQ DI container

    compile(...) arguments:
        plan_node: Callable (state -> state). Produces next thought / tool requests
        tool_node: ToolNode executing declared tools
        reflect_node: Callable (state -> state). Consumes tool results & may adjust plan
        condition: Optional Callable[[AgentState], str] returning next node name or END
        checkpointer/store/interrupt_before/interrupt_after/callback_manager:
            Standard graph compilation options

    Returns:
        CompiledGraph ready for invoke / ainvoke.

    Notes:
        - Node names can be customized via (callable, "NAME") tuples.
        - condition must return one of: tool_node_name, reflect_node_name, END.
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
        plan_node: Callable | tuple[Callable, str],
        tool_node: ToolNode | tuple[ToolNode, str],
        reflect_node: Callable | tuple[Callable, str],
        *,
        condition: Callable[[AgentState], str] | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        """Compile the Plan-Act-Reflect loop.

        Args:
            plan_node: Callable or (callable, name)
            tool_node: ToolNode or (ToolNode, name)
            reflect_node: Callable or (callable, name)
            condition: Optional decision function. Defaults to internal heuristic.
            checkpointer/store/interrupt_* / callback_manager: Standard graph options.

        Returns:
            CompiledGraph
        """
        # PLAN
        if isinstance(plan_node, tuple):
            plan_func, plan_name = plan_node
            if not callable(plan_func):
                raise ValueError("plan_node[0] must be callable")
        else:
            plan_func = plan_node
            plan_name = "PLAN"
            if not callable(plan_func):
                raise ValueError("plan_node must be callable")

        # ACT
        if isinstance(tool_node, tuple):
            tool_func, tool_name = tool_node
            if not isinstance(tool_func, ToolNode):
                raise ValueError("tool_node[0] must be a ToolNode")
        else:
            tool_func = tool_node
            tool_name = "ACT"
            if not isinstance(tool_func, ToolNode):
                raise ValueError("tool_node must be a ToolNode")

        # REFLECT
        if isinstance(reflect_node, tuple):
            reflect_func, reflect_name = reflect_node
            if not callable(reflect_func):
                raise ValueError("reflect_node[0] must be callable")
        else:
            reflect_func = reflect_node
            reflect_name = "REFLECT"
            if not callable(reflect_func):
                raise ValueError("reflect_node must be callable")

        # Register nodes
        self._graph.add_node(plan_name, plan_func)
        self._graph.add_node(tool_name, tool_func)
        self._graph.add_node(reflect_name, reflect_func)

        # Decision
        decision_fn = condition or _should_act
        self._graph.add_conditional_edges(
            plan_name,
            decision_fn,
            {tool_name: tool_name, reflect_name: reflect_name, END: END},
        )

        # Loop edges
        self._graph.add_edge(tool_name, reflect_name)
        self._graph.add_edge(reflect_name, plan_name)

        # Entry
        self._graph.set_entry_point(plan_name)

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
