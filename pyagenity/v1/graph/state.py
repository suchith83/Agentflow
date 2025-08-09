from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid
import time


class SessionStatus:
    RUNNING = "RUNNING"
    WAITING_HUMAN = "WAITING_HUMAN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class GraphState:
    session_id: str
    current_node: Optional[str]
    status: str = SessionStatus.RUNNING
    shared: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def clone(self) -> "GraphState":
        return GraphState(
            session_id=self.session_id,
            current_node=self.current_node,
            status=self.status,
            shared=dict(self.shared),
            messages=list(self.messages),
            created_at=self.created_at,
            updated_at=time.time(),
        )


class InMemoryStateStore:
    """Simple in-memory state store for sessions. Not thread-safe."""

    def __init__(self):
        self._sessions: Dict[str, GraphState] = {}

    def create(self, start_node: Optional[str]) -> GraphState:
        sid = str(uuid.uuid4())
        state = GraphState(session_id=sid, current_node=start_node)
        self._sessions[sid] = state
        return state

    def get(self, session_id: str) -> Optional[GraphState]:
        return self._sessions.get(session_id)

    def update(self, state: GraphState) -> None:
        state.updated_at = time.time()
        self._sessions[state.session_id] = state

    def list(self) -> List[GraphState]:
        return list(self._sessions.values())
