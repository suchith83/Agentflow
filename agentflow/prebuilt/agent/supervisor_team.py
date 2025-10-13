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

    def compile(  # noqa: PLR0912
        self,
        supervisor_node: Callable | tuple[Callable, str],
        workers: dict[str, Callable | ToolNode | tuple[Callable | ToolNode, str]],
        condition: Callable[[AgentState], str],
        aggregate_node: Callable | tuple[Callable, str] | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        # Handle supervisor_node
        if isinstance(supervisor_node, tuple):
            supervisor_func, supervisor_name = supervisor_node
            if not callable(supervisor_func):
                raise ValueError("supervisor_node[0] must be callable")
        else:
            supervisor_func = supervisor_node
            supervisor_name = "SUPERVISOR"
            if not callable(supervisor_func):
                raise ValueError("supervisor_node must be callable")

        if not workers:
            raise ValueError("workers must be a non-empty dict")

        # Add worker nodes
        worker_names = []
        for key, fn in workers.items():
            if isinstance(fn, tuple):
                worker_func, worker_name = fn
                if not (callable(worker_func) or isinstance(worker_func, ToolNode)):
                    raise ValueError(f"Worker '{key}'[0] must be callable or ToolNode")
            else:
                worker_func = fn
                worker_name = key
                if not (callable(worker_func) or isinstance(worker_func, ToolNode)):
                    raise ValueError(f"Worker '{key}' must be callable or ToolNode")
            self._graph.add_node(worker_name, worker_func)
            worker_names.append(worker_name)

        # Handle aggregate_node
        aggregate_name = "AGGREGATE"
        if aggregate_node:
            if isinstance(aggregate_node, tuple):
                aggregate_func, aggregate_name = aggregate_node
                if not callable(aggregate_func):
                    raise ValueError("aggregate_node[0] must be callable")
            else:
                aggregate_func = aggregate_node
                aggregate_name = "AGGREGATE"
                if not callable(aggregate_func):
                    raise ValueError("aggregate_node must be callable")
            self._graph.add_node(aggregate_name, aggregate_func)

        # SUPERVISOR decides next worker
        path_map = {k: k for k in worker_names}
        path_map[END] = END
        self._graph.add_conditional_edges(supervisor_name, condition, path_map)

        # After worker, go to AGGREGATE if present, else back to SUPERVISOR
        for name in worker_names:
            self._graph.add_edge(name, aggregate_name if aggregate_node else supervisor_name)

        if aggregate_node:
            self._graph.add_edge(aggregate_name, supervisor_name)

        self._graph.set_entry_point(supervisor_name)

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
