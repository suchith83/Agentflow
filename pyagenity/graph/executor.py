from __future__ import annotations
from typing import Optional, Dict, Any, Callable, List
from .graph import Graph
from .state import InMemoryStateStore, GraphState, SessionStatus
from .nodes import HumanInputNode

FinalHook = Callable[[GraphState], None]


class GraphExecutor:
    def __init__(self, graph: Graph, store: Optional[InMemoryStateStore] = None):
        self.graph = graph
        self.store = store or InMemoryStateStore()
        self._final_hooks: List[FinalHook] = []
        self.graph.validate()

    def add_final_hook(self, hook: FinalHook):
        self._final_hooks.append(hook)

    def start(self, initial_shared: Optional[Dict[str, Any]] = None) -> GraphState:
        state = self.store.create(self.graph.start_node)
        if initial_shared:
            state.shared.update(initial_shared)
        self.store.update(state)
        return self._run(state)

    def resume(self, session_id: str, human_input: Optional[str] = None) -> GraphState:
        state = self.store.get(session_id)
        if state is None:
            raise ValueError("Unknown session id")
        if human_input is not None:
            state.shared["human_input"] = human_input
        if state.status != SessionStatus.WAITING_HUMAN:
            raise ValueError("Session not waiting for human input")
        state.status = SessionStatus.RUNNING
        self.store.update(state)
        return self._run(state)

    def _run(self, state: GraphState) -> GraphState:
        while state.status == SessionStatus.RUNNING and state.current_node is not None:
            node = self.graph.nodes.get(state.current_node)
            if node is None:
                state.status = SessionStatus.FAILED
                break
            try:
                node.run(state.shared)
            except Exception:
                state.status = SessionStatus.FAILED
                break
            # Human input pause
            if (
                isinstance(node, HumanInputNode)
                and state.shared.get("human_input") is None
            ):
                state.status = SessionStatus.WAITING_HUMAN
                break
            # Determine next nodes
            next_list = self.graph.next_nodes(state.current_node, state.shared)
            if not next_list:
                state.status = SessionStatus.COMPLETED
                break
            # For now choose first deterministically
            state.current_node = next_list[0]
            self.store.update(state)
        self.store.update(state)
        if state.status in (SessionStatus.COMPLETED, SessionStatus.FAILED):
            for hook in self._final_hooks:
                try:
                    hook(state)
                except Exception:
                    pass
        return state
