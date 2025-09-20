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

    def compile(
        self,
        map_node: Callable | ToolNode,
        reduce_node: Callable,
        split_node: Callable | None = None,
        condition: Callable[[AgentState], str] | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        # Add nodes
        if split_node:
            self._graph.add_node("SPLIT", split_node)
        if not (callable(map_node) or isinstance(map_node, ToolNode)):
            raise ValueError("map_node must be callable or ToolNode")
        self._graph.add_node("MAP", map_node)  # type: ignore[arg-type]
        if not callable(reduce_node):
            raise ValueError("reduce_node must be callable")
        self._graph.add_node("REDUCE", reduce_node)

        # Edges
        if split_node:
            self._graph.add_edge("SPLIT", "MAP")
            self._graph.set_entry_point("SPLIT")
        else:
            self._graph.set_entry_point("MAP")

        self._graph.add_edge("MAP", "REDUCE")

        # Continue mapping or finish
        if condition is None:
            # default: finish after one map-reduce
            def _cond(_: AgentState) -> str:
                return END

            condition = _cond

        self._graph.add_conditional_edges("REDUCE", condition, {"MAP": "MAP", END: END})

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
