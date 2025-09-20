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


class RAGAgent[StateT: AgentState]:
    """Simple RAG: retrieve -> synthesize; optional follow-up.

    Nodes:
    - RETRIEVE: uses a retriever (callable or ToolNode) to fetch context
    - SYNTHESIZE: LLM/composer builds an answer
    - Optional condition: loop back to RETRIEVE for follow-up queries; else END
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
        retriever_node: Callable | ToolNode,
        synthesize_node: Callable,
        followup_condition: Callable[[AgentState], str] | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        # Nodes
        if not (callable(retriever_node) or isinstance(retriever_node, ToolNode)):
            raise ValueError("retriever_node must be callable or ToolNode")
        self._graph.add_node("RETRIEVE", retriever_node)  # type: ignore[arg-type]
        if not callable(synthesize_node):
            raise ValueError("synthesize_node must be callable")
        self._graph.add_node("SYNTHESIZE", synthesize_node)

        # Edges
        self._graph.add_edge("RETRIEVE", "SYNTHESIZE")
        self._graph.set_entry_point("RETRIEVE")

        if followup_condition is None:
            # default: END after synthesize
            def _cond(_: AgentState) -> str:
                return END

            followup_condition = _cond

        self._graph.add_conditional_edges(
            "SYNTHESIZE",
            followup_condition,
            {"RETRIEVE": "RETRIEVE", END: END},
        )

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )
