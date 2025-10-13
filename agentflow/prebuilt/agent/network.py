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
from agentflow.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


StateT = TypeVar("StateT", bound=AgentState)


class NetworkAgent[StateT: AgentState]:
    """Network pattern: define arbitrary node set and routing policies.

    - Nodes can be callables or ToolNode.
    - Edges can be static or conditional via a router function per node.
    - Entry point is explicit.
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
        nodes: dict[str, Callable | ToolNode | tuple[Callable | ToolNode, str]],
        entry: str,
        static_edges: list[tuple[str, str]] | None = None,
        conditional_edges: list[tuple[str, Callable[[AgentState], str], dict[str, str]]]
        | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        if not nodes:
            raise ValueError("nodes must be a non-empty dict")

        # Add nodes
        for key, fn in nodes.items():
            if isinstance(fn, tuple):
                func, name = fn
            else:
                func, name = fn, key
            if not (callable(func) or isinstance(func, ToolNode)):
                raise ValueError(f"Node '{key}' must be a callable or ToolNode")
            self._graph.add_node(name, func)

        if entry not in self._graph.nodes:
            raise ValueError(f"entry node '{entry}' must be present in nodes")

        # Static edges
        for src, dst in static_edges or []:
            if src not in self._graph.nodes or dst not in self._graph.nodes:
                raise ValueError(f"Invalid static edge {src}->{dst}: unknown node")
            self._graph.add_edge(src, dst)

        # Conditional edges
        for src, cond, pmap in conditional_edges or []:
            if src not in self._graph.nodes:
                raise ValueError(f"Invalid conditional edge: unknown node '{src}'")
            self._graph.add_conditional_edges(src, cond, pmap)

        # Note: callers may include END in path maps; not enforced here.

        self._graph.set_entry_point(entry)

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
