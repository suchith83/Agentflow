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


class RouterAgent[StateT: AgentState]:
    """A configurable router-style agent.

    Pattern:
    - A router node runs (LLM or custom logic) and may update state/messages
    - A condition function inspects the state and returns a route key
    - Edges route to the matching node; each route returns back to ROUTER
    - Return END (via condition) to finish

    Usage:
        router = RouterAgent()
        app = router.compile(
            router_node=my_router_func,  # def my_router_func(state, config, ...)
            routes={
                "search": search_node,
                "summarize": summarize_node,
            },
            # Condition inspects state and returns one of the keys above or END
            condition=my_condition,  # def my_condition(state) -> str
            # Optional explicit path map if returned keys differ from node names
            # path_map={"SEARCH": "search", "SUM": "summarize", END: END}
        )
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
        router_node: Callable | tuple[Callable, str],
        routes: dict[str, Callable | ToolNode | tuple[Callable | ToolNode, str]],
        condition: Callable[[AgentState], str] | None = None,
        path_map: dict[str, str] | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        # Handle router_node
        if isinstance(router_node, tuple):
            router_func, router_name = router_node
            if not callable(router_func):
                raise ValueError("router_node[0] must be callable")
        else:
            router_func = router_node
            router_name = "ROUTER"
            if not callable(router_func):
                raise ValueError("router_node must be callable")

        if not routes:
            raise ValueError("routes must be a non-empty dict of name -> callable/ToolNode/tuple")

        # Add route nodes
        route_names = []
        for key, func in routes.items():
            if isinstance(func, tuple):
                route_func, route_name = func
                if not (callable(route_func) or isinstance(route_func, ToolNode)):
                    raise ValueError(f"Route '{key}'[0] must be callable or ToolNode")
            else:
                route_func = func
                route_name = key
                if not (callable(route_func) or isinstance(route_func, ToolNode)):
                    raise ValueError(f"Route '{key}' must be callable or ToolNode")
            self._graph.add_node(route_name, route_func)
            route_names.append(route_name)

        # Add router node as entry
        self._graph.add_node(router_name, router_func)

        # Build default condition/path_map if needed
        if condition is None and len(route_names) == 1:
            only = route_names[0]

            def _always(_: AgentState) -> str:
                return only

            condition = _always
            path_map = {only: only, END: END}

        if condition is None and len(route_names) > 1:
            raise ValueError("condition must be provided when multiple routes are defined")

        # If path_map is not provided, assume router returns exact route names
        if path_map is None:
            path_map = {k: k for k in route_names}
            path_map[END] = END

        # Conditional edges from router node based on condition results
        self._graph.add_conditional_edges(
            router_name,
            condition,  # type: ignore[arg-type]
            path_map,
        )

        # Loop back to router node from each route node
        for name in route_names:
            self._graph.add_edge(name, router_name)

        # Entry
        self._graph.set_entry_point(router_name)

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
