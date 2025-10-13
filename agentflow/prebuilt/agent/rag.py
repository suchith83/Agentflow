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
        retriever_node: Callable | ToolNode | tuple[Callable | ToolNode, str],
        synthesize_node: Callable | tuple[Callable, str],
        followup_condition: Callable[[AgentState], str] | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        # Nodes
        # Handle retriever_node
        if isinstance(retriever_node, tuple):
            retriever_func, retriever_name = retriever_node
            if not (callable(retriever_func) or isinstance(retriever_func, ToolNode)):
                raise ValueError("retriever_node[0] must be callable or ToolNode")
        else:
            retriever_func = retriever_node
            retriever_name = "RETRIEVE"
            if not (callable(retriever_func) or isinstance(retriever_func, ToolNode)):
                raise ValueError("retriever_node must be callable or ToolNode")

        # Handle synthesize_node
        if isinstance(synthesize_node, tuple):
            synthesize_func, synthesize_name = synthesize_node
            if not callable(synthesize_func):
                raise ValueError("synthesize_node[0] must be callable")
        else:
            synthesize_func = synthesize_node
            synthesize_name = "SYNTHESIZE"
            if not callable(synthesize_func):
                raise ValueError("synthesize_node must be callable")

        self._graph.add_node(retriever_name, retriever_func)  # type: ignore[arg-type]
        self._graph.add_node(synthesize_name, synthesize_func)

        # Edges
        self._graph.add_edge(retriever_name, synthesize_name)
        self._graph.set_entry_point(retriever_name)

        if followup_condition is None:
            # default: END after synthesize
            def _cond(_: AgentState) -> str:
                return END

            followup_condition = _cond

        self._graph.add_conditional_edges(
            synthesize_name,
            followup_condition,
            {retriever_name: retriever_name, END: END},
        )

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )

    def compile_advanced(
        self,
        retriever_nodes: list[Callable | ToolNode | tuple[Callable | ToolNode, str]],
        synthesize_node: Callable | tuple[Callable, str],
        options: dict | None = None,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
    ) -> CompiledGraph:
        """Advanced RAG wiring with hybrid retrieval and optional stages.

        Chain:
          (QUERY_PLAN?) -> R1 -> (MERGE?) -> R2 -> (MERGE?) -> ...
          -> (RERANK?) -> (COMPRESS?) -> SYNTHESIZE -> cond
        Each retriever may be a different modality (sparse, dense, self-query, MMR, etc.).
        """

        options = options or {}
        query_plan_node = options.get("query_plan")
        merger_node = options.get("merge")
        rerank_node = options.get("rerank")
        compress_node = options.get("compress")
        followup_condition = options.get("followup_condition")

        qname = self._add_optional_node(
            query_plan_node,
            default_name="QUERY_PLAN",
            label="query_plan",
        )

        # Add retrievers
        r_names = self._add_retriever_nodes(retriever_nodes)

        # Optional stages
        mname = self._add_optional_node(merger_node, default_name="MERGE", label="merge")
        rrname = self._add_optional_node(rerank_node, default_name="RERANK", label="rerank")
        cname = self._add_optional_node(
            compress_node,
            default_name="COMPRESS",
            label="compress",
        )

        # Synthesize
        sname = self._add_synthesize_node(synthesize_node)

        # Wire edges end-to-end and follow-up
        self._wire_advanced_edges(qname, r_names, mname, rrname, cname, sname, followup_condition)

        return self._graph.compile(
            checkpointer=checkpointer,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            callback_manager=callback_manager,
        )

    # ---- helpers ----
    def _add_optional_node(
        self,
        node: Callable | tuple[Callable, str] | None,
        *,
        default_name: str,
        label: str,
    ) -> str | None:
        if not node:
            return None
        if isinstance(node, tuple):
            func, name = node
        else:
            func, name = node, default_name
        if not callable(func):
            raise ValueError(f"{label} node must be callable")
        self._graph.add_node(name, func)
        return name

    def _add_retriever_nodes(
        self,
        retriever_nodes: list[Callable | ToolNode | tuple[Callable | ToolNode, str]],
    ) -> list[str]:
        if not retriever_nodes:
            raise ValueError("retriever_nodes must be non-empty")
        names: list[str] = []
        for idx, rn in enumerate(retriever_nodes):
            if isinstance(rn, tuple):
                rfunc, rname = rn
            else:
                rfunc, rname = rn, f"RETRIEVE_{idx + 1}"
            if not (callable(rfunc) or isinstance(rfunc, ToolNode)):
                raise ValueError("retriever must be callable or ToolNode")
            self._graph.add_node(rname, rfunc)  # type: ignore[arg-type]
            names.append(rname)
        return names

    def _add_synthesize_node(self, synthesize_node: Callable | tuple[Callable, str]) -> str:
        if isinstance(synthesize_node, tuple):
            sfunc, sname = synthesize_node
        else:
            sfunc, sname = synthesize_node, "SYNTHESIZE"
        if not callable(sfunc):
            raise ValueError("synthesize_node must be callable")
        self._graph.add_node(sname, sfunc)
        return sname

    def _wire_advanced_edges(
        self,
        qname: str | None,
        r_names: list[str],
        mname: str | None,
        rrname: str | None,
        cname: str | None,
        sname: str,
        followup_condition: Callable[[AgentState], str] | None = None,
    ) -> None:
        entry = qname or r_names[0]
        self._graph.set_entry_point(entry)
        if qname:
            self._graph.add_edge(qname, r_names[0])

        tail_target = rrname or cname or sname
        for i, rname in enumerate(r_names):
            is_last = i == len(r_names) - 1
            nxt = r_names[i + 1] if not is_last else tail_target
            if mname:
                self._graph.add_edge(rname, mname)
                self._graph.add_edge(mname, nxt)
            else:
                self._graph.add_edge(rname, nxt)

        # Tail wiring
        if rrname and cname:
            self._graph.add_edge(rrname, cname)
            self._graph.add_edge(cname, sname)
        elif rrname:
            self._graph.add_edge(rrname, sname)
        elif cname:
            self._graph.add_edge(cname, sname)

        # default follow-up to END
        if followup_condition is None:

            def _cond(_: AgentState) -> str:
                return END

            followup_condition = _cond

        entry_node = qname or r_names[0]
        path_map = {entry_node: entry_node, END: END}
        self._graph.add_conditional_edges(sname, followup_condition, path_map)
