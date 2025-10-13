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


class SwarmAgent[StateT: AgentState]:
    """Swarm pattern: dispatch to many workers, collect, then reach consensus.

    Notes:
    - The underlying engine executes nodes sequentially; true parallelism isn't
      performed at the graph level. For concurrency, worker/collector nodes can
      internally use BackgroundTaskManager or async to fan-out.
    - This pattern wires a linear broadcast-collect chain ending in CONSENSUS.

    Nodes:
    - optional DISPATCH: prepare/plan the swarm task
    - WORKER_i: a set of worker nodes (Callable or ToolNode)
    - optional COLLECT: consolidate each worker's result into shared state
    - CONSENSUS: aggregate all collected results and produce final output
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
        workers: dict[str, Callable | ToolNode | tuple[Callable | ToolNode, str]],
        consensus_node: Callable | tuple[Callable, str],
        options: dict | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        resolved_workers = self._add_worker_nodes(workers)
        worker_sequence = resolved_workers

        options = options or {}
        dispatch_node = options.get("dispatch")
        collect_node = options.get("collect")
        followup_condition = options.get("followup_condition")

        dispatch_name = self._resolve_dispatch(dispatch_node)
        collect_info = self._resolve_collect(collect_node)
        consensus_name = self._resolve_consensus(consensus_node)

        entry = dispatch_name or worker_sequence[0]
        self._graph.set_entry_point(entry)
        if dispatch_name:
            self._graph.add_edge(dispatch_name, worker_sequence[0])

        self._wire_edges(worker_sequence, collect_info, consensus_name)

        if followup_condition is None:

            def _cond(_: AgentState) -> str:
                return END

            followup_condition = _cond

        self._graph.add_conditional_edges(
            consensus_name,
            followup_condition,
            {entry: entry, END: END},
        )

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )

    # ---- helpers ----
    def _add_worker_nodes(
        self,
        workers: dict[str, Callable | ToolNode | tuple[Callable | ToolNode, str]],
    ) -> list[str]:
        if not workers:
            raise ValueError("workers must be a non-empty dict")

        names: list[str] = []
        for key, fn in workers.items():
            if isinstance(fn, tuple):
                func, name = fn
            else:
                func, name = fn, key
            if not (callable(func) or isinstance(func, ToolNode)):
                raise ValueError(f"Worker '{key}' must be a callable or ToolNode")
            self._graph.add_node(name, func)
            names.append(name)
        return names

    def _resolve_dispatch(self, node: Callable | tuple[Callable, str] | None) -> str | None:
        if not node:
            return None
        if isinstance(node, tuple):
            func, name = node
        else:
            func, name = node, "DISPATCH"
        if not callable(func):
            raise ValueError("dispatch node must be callable")
        self._graph.add_node(name, func)
        return name

    def _resolve_collect(
        self,
        node: Callable | tuple[Callable, str] | None,
    ) -> tuple[Callable, str] | None:
        if not node:
            return None
        if isinstance(node, tuple):
            func, name = node
        else:
            func, name = node, "COLLECT"
        if not callable(func):
            raise ValueError("collect node must be callable")
        # Do not add a single shared collect node to avoid ambiguous routing.
        # We'll create per-worker collect nodes during wiring using this (func, base_name).
        return func, name

    def _resolve_consensus(self, node: Callable | tuple[Callable, str]) -> str:
        if isinstance(node, tuple):
            func, name = node
        else:
            func, name = node, "CONSENSUS"
        if not callable(func):
            raise ValueError("consensus node must be callable")
        self._graph.add_node(name, func)
        return name

    def _wire_edges(
        self,
        worker_sequence: list[str],
        collect_info: tuple[Callable, str] | None,
        consensus_name: str,
    ) -> None:
        for i, wname in enumerate(worker_sequence):
            is_last = i == len(worker_sequence) - 1
            target = consensus_name if is_last else worker_sequence[i + 1]
            if collect_info:
                cfunc, base = collect_info
                cname = f"{base}_{i + 1}"
                # Create a dedicated collect node per worker to prevent loops
                self._graph.add_node(cname, cfunc)
                self._graph.add_edge(wname, cname)
                self._graph.add_edge(cname, target)
            else:
                self._graph.add_edge(wname, target)
