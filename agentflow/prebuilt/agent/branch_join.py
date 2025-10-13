from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from injectq import InjectQ

from agentflow.checkpointer.base_checkpointer import BaseCheckpointer
from agentflow.graph.compiled_graph import CompiledGraph
from agentflow.graph.state_graph import StateGraph
from agentflow.publisher.base_publisher import BasePublisher
from agentflow.state.agent_state import AgentState
from agentflow.state.base_context import BaseContextManager
from agentflow.store.base_store import BaseStore
from agentflow.utils.callbacks import CallbackManager
from agentflow.utils.constants import END
from agentflow.utils.id_generator import BaseIDGenerator, DefaultIDGenerator


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
        branches: dict[str, Callable | tuple[Callable, str]],
        join_node: Callable | tuple[Callable, str],
        next_branch_condition: Callable | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        if not branches:
            raise ValueError("branches must be a non-empty dict of name -> callable/tuple")

        # Add branch nodes
        branch_names = []
        for key, fn in branches.items():
            if isinstance(fn, tuple):
                branch_func, branch_name = fn
                if not callable(branch_func):
                    raise ValueError(f"Branch '{key}'[0] must be callable")
            else:
                branch_func = fn
                branch_name = key
                if not callable(branch_func):
                    raise ValueError(f"Branch '{key}' must be callable")
            self._graph.add_node(branch_name, branch_func)
            branch_names.append(branch_name)

        # Handle join_node
        if isinstance(join_node, tuple):
            join_func, join_name = join_node
            if not callable(join_func):
                raise ValueError("join_node[0] must be callable")
        else:
            join_func = join_node
            join_name = "JOIN"
            if not callable(join_func):
                raise ValueError("join_node must be callable")
        self._graph.add_node(join_name, join_func)

        # Wire branches to JOIN
        for name in branch_names:
            self._graph.add_edge(name, join_name)

        # Entry: first branch
        first = branch_names[0]
        self._graph.set_entry_point(first)

        # Decide next branch or END after join
        if next_branch_condition is None:
            # default: END after join
            def _cond(_: AgentState) -> str:
                return END

            next_branch_condition = _cond

        # next_branch_condition returns a branch name or END
        path_map = {k: k for k in branch_names}
        path_map[END] = END
        self._graph.add_conditional_edges(join_name, next_branch_condition, path_map)

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
