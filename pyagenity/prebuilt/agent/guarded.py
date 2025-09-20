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


def _guard_condition_factory(
    validator: Callable[[AgentState], bool],
    max_attempts: int,
):
    attempts_key = "_guard_attempts"

    def _cond(state: AgentState) -> str:
        # maintain attempt count in execution_meta.data
        data = getattr(state.execution_meta, "internal_data", None) or {}
        attempts = data.get(attempts_key, 0)
        is_valid = validator(state)
        if is_valid:
            return END
        attempts += 1
        data[attempts_key] = attempts
        state.execution_meta.internal_data = data
        return "REPAIR" if attempts <= max_attempts else END

    return _cond


class GuardedAgent[StateT: AgentState]:
    """Validate output and repair until valid or attempts exhausted.

    Nodes:
    - PRODUCE: main generation node
    - REPAIR: correction node when validation fails

    Edges:
    PRODUCE -> conditional(valid? END : REPAIR)
    REPAIR -> PRODUCE
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
        produce_node: Callable,
        repair_node: Callable,
        validator: Callable[[AgentState], bool],
        max_attempts: int = 2,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        if not callable(produce_node) or not callable(repair_node):
            raise ValueError("produce_node and repair_node must be callable")

        self._graph.add_node("PRODUCE", produce_node)
        self._graph.add_node("REPAIR", repair_node)

        # produce -> END or REPAIR
        condition = _guard_condition_factory(validator, max_attempts)
        self._graph.add_conditional_edges("PRODUCE", condition, {"REPAIR": "REPAIR", END: END})
        # repair -> produce
        self._graph.add_edge("REPAIR", "PRODUCE")

        self._graph.set_entry_point("PRODUCE")

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
