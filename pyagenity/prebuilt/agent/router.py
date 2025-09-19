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

    def compile(
        self,
        router_node: Callable,
        routes: dict[str, Callable | ToolNode],
        condition: Callable[[AgentState], str] | None = None,
        path_map: dict[str, str] | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        if not callable(router_node):
            raise ValueError("router_node must be a callable function")

        if not routes:
            raise ValueError("routes must be a non-empty dict of name -> callable/ToolNode")

        # Add ROUTER as entry
        self._graph.add_node("ROUTER", router_node)

        # Add route nodes
        for name, func in routes.items():
            if not (callable(func) or isinstance(func, ToolNode)):
                raise ValueError(f"Route '{name}' must be a callable or ToolNode")
            self._graph.add_node(name, func)  # type: ignore[arg-type]

        # Build default condition/path_map if needed
        if condition is None and len(routes) == 1:
            # If there's only one route, always choose it
            only = next(iter(routes.keys()))

            def _always(_: AgentState) -> str:
                return only

            condition = _always
            path_map = {only: only, END: END}

        if condition is None and len(routes) > 1:
            raise ValueError("condition must be provided when multiple routes are defined")

        # If path_map is not provided, assume router returns exact route names
        if path_map is None:
            path_map = {k: k for k in routes}
            path_map[END] = END

        # Conditional edges from ROUTER based on condition results
        self._graph.add_conditional_edges(
            "ROUTER",
            condition,  # type: ignore[arg-type]
            path_map,
        )

        # Loop back to ROUTER from each route node
        for name in routes:
            self._graph.add_edge(name, "ROUTER")

        # Entry
        self._graph.set_entry_point("ROUTER")

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
