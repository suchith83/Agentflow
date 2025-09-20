from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from injectq import InjectQ

from pyagenity.checkpointer.base_checkpointer import BaseCheckpointer
from pyagenity.graph.compiled_graph import CompiledGraph
from pyagenity.graph.state_graph import StateGraph
from pyagenity.publisher.base_publisher import BasePublisher
from pyagenity.state.agent_state import AgentState
from pyagenity.state.base_context import BaseContextManager
from pyagenity.store.base_store import BaseStore
from pyagenity.utils.callbacks import CallbackManager
from pyagenity.utils.constants import END
from pyagenity.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


StateT = TypeVar("StateT", bound=AgentState)


class BranchJoinAgent[StateT: AgentState]:
    """Execute multiple branches then join.

    Note: This prebuilt models branches sequentially (not true parallel execution).
    For each provided branch node, we add edges branch_i -> JOIN. The JOIN node
    decides whether more branches remain or END. A more advanced version could
    use BackgroundTaskManager for concurrency.
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
        branches: dict[str, Callable],
        join_node: Callable,
        next_branch_condition: Callable[[AgentState], str] | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        if not branches:
            raise ValueError("branches must be a non-empty dict of name -> callable")
        if not callable(join_node):
            raise ValueError("join_node must be callable")

        # Add nodes
        for name, fn in branches.items():
            if not callable(fn):
                raise ValueError(f"Branch '{name}' must be callable")
            self._graph.add_node(name, fn)

        self._graph.add_node("JOIN", join_node)

        # Wire branches to JOIN
        for name in branches:
            self._graph.add_edge(name, "JOIN")

        # Entry: first branch
        first = next(iter(branches.keys()))
        self._graph.set_entry_point(first)

        # Decide next branch or END after join
        if next_branch_condition is None:
            # default: END after join
            def _cond(_: AgentState) -> str:
                return END

            next_branch_condition = _cond

        # next_branch_condition returns a branch name or END
        path_map = {k: k for k in branches}
        path_map[END] = END
        self._graph.add_conditional_edges("JOIN", next_branch_condition, path_map)

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
