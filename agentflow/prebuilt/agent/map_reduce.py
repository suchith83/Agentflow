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


class MapReduceAgent[StateT: AgentState]:
    """Map over items then reduce.

    Nodes:
    - SPLIT: optional, prepares per-item tasks (or state already contains items)
    - MAP: processes one item per iteration
    - REDUCE: aggregates results and decides END or continue

    Compile requires:
      map_node: Callable|ToolNode
      reduce_node: Callable
      split_node: Callable | None
      condition: Callable[[AgentState], str] returns "MAP" to continue or END
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

    def compile(  # noqa: PLR0912
        self,
        map_node: Callable | ToolNode | tuple[Callable | ToolNode, str],
        reduce_node: Callable | tuple[Callable, str],
        split_node: Callable | tuple[Callable, str] | None = None,
        condition: Callable[[AgentState], str] | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        # Handle split_node
        split_name = "SPLIT"
        if split_node:
            if isinstance(split_node, tuple):
                split_func, split_name = split_node
                if not callable(split_func):
                    raise ValueError("split_node[0] must be callable")
            else:
                split_func = split_node
                split_name = "SPLIT"
                if not callable(split_func):
                    raise ValueError("split_node must be callable")
            self._graph.add_node(split_name, split_func)

        # Handle map_node
        if isinstance(map_node, tuple):
            map_func, map_name = map_node
            if not (callable(map_func) or isinstance(map_func, ToolNode)):
                raise ValueError("map_node[0] must be callable or ToolNode")
        else:
            map_func = map_node
            map_name = "MAP"
            if not (callable(map_func) or isinstance(map_func, ToolNode)):
                raise ValueError("map_node must be callable or ToolNode")
        self._graph.add_node(map_name, map_func)

        # Handle reduce_node
        if isinstance(reduce_node, tuple):
            reduce_func, reduce_name = reduce_node
            if not callable(reduce_func):
                raise ValueError("reduce_node[0] must be callable")
        else:
            reduce_func = reduce_node
            reduce_name = "REDUCE"
            if not callable(reduce_func):
                raise ValueError("reduce_node must be callable")
        self._graph.add_node(reduce_name, reduce_func)

        # Edges
        if split_node:
            self._graph.add_edge(split_name, map_name)
            self._graph.set_entry_point(split_name)
        else:
            self._graph.set_entry_point(map_name)

        self._graph.add_edge(map_name, reduce_name)

        # Continue mapping or finish
        if condition is None:
            # default: finish after one map-reduce
            def _cond(_: AgentState) -> str:
                return END

            condition = _cond

        self._graph.add_conditional_edges(reduce_name, condition, {map_name: map_name, END: END})

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
