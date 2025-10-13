from __future__ import annotations

from collections.abc import Callable, Sequence
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


class SequentialAgent[StateT: AgentState]:
    """A simple sequential agent that executes a fixed pipeline of nodes.

    Pattern:
    - Nodes run in the provided order: step1 -> step2 -> ... -> stepN
    - After the last step, the graph ends

    Usage:
        seq = SequentialAgent()
        app = seq.compile([
            ("ingest", ingest_node),
            ("plan", plan_node),
            ("execute", execute_node),
        ])
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
        steps: Sequence[tuple[str, Callable | ToolNode] | tuple[Callable | ToolNode, str]],
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        if not steps or len(steps) == 0:
            raise ValueError(
                "steps must be a non-empty sequence of (name, callable/ToolNode) o"
                "or (callable/ToolNode, name)"
            )

        # Add nodes
        step_names = []
        for step in steps:
            if isinstance(step[0], str):
                name, func = step
            else:
                func, name = step
            if not (callable(func) or isinstance(func, ToolNode)):
                raise ValueError(f"Step '{name}' must be a callable or ToolNode")
            self._graph.add_node(name, func)  # type: ignore[arg-type]
            step_names.append(name)

        # Static edges in order
        for i in range(len(step_names) - 1):
            self._graph.add_edge(step_names[i], step_names[i + 1])

        # Entry is the first step
        self._graph.set_entry_point(step_names[0])

        # No explicit edge to END needed; the engine will end if no outgoing edges remain.
        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
