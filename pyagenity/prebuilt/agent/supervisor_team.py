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


class SupervisorTeamAgent[StateT: AgentState]:
    """Supervisor routes tasks to worker nodes and aggregates results.

    Nodes:
    - SUPERVISOR: decides which worker to call (by returning a worker key) or END
    - Multiple WORKER nodes: functions or ToolNode instances
    - AGGREGATE: optional aggregator node after worker runs; loops back to SUPERVISOR

    The compile requires:
      supervisor_node: Callable
      workers: dict[str, Callable|ToolNode]
      aggregate_node: Callable | None
      condition: Callable[[AgentState], str] returns worker key or END
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
        supervisor_node: Callable,
        workers: dict[str, Callable | ToolNode],
        condition: Callable[[AgentState], str],
        aggregate_node: Callable | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        if not callable(supervisor_node):
            raise ValueError("supervisor_node must be callable")
        if not workers:
            raise ValueError("workers must be a non-empty dict")

        self._graph.add_node("SUPERVISOR", supervisor_node)
        for name, fn in workers.items():
            if not (callable(fn) or isinstance(fn, ToolNode)):
                raise ValueError(f"Worker '{name}' must be callable or ToolNode")
            self._graph.add_node(name, fn)  # type: ignore[arg-type]

        if aggregate_node:
            self._graph.add_node("AGGREGATE", aggregate_node)

        # SUPERVISOR decides next worker
        path_map = {k: k for k in workers}
        path_map[END] = END
        self._graph.add_conditional_edges("SUPERVISOR", condition, path_map)

        # After worker, go to AGGREGATE if present, else back to SUPERVISOR
        for name in workers:
            self._graph.add_edge(name, "AGGREGATE" if aggregate_node else "SUPERVISOR")

        if aggregate_node:
            self._graph.add_edge("AGGREGATE", "SUPERVISOR")

        self._graph.set_entry_point("SUPERVISOR")

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
