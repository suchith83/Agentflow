from typing import Any, Dict, List, Optional
from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import Message
from .base_checkpointer import BaseCheckpointer


class InMemoryCheckpointer(BaseCheckpointer):
    def __init__(self):
        # Simulate tables with dicts/lists
        self._threads: Dict[str, Dict[str, Any]] = {}
        self._messages: Dict[str, List[Message]] = {}
        self._states: Dict[str, AgentState] = {}

    def put(
        self,
        config: Dict[str, Any],
        messages: list[Message],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._messages[thread_id] = messages.copy()
        if metadata:
            self._threads.setdefault(thread_id, {})["metadata"] = metadata

    def get(self, config: Dict[str, Any]) -> list[Message]:
        thread_id = config.get("thread_id", "default")
        return self._messages.get(thread_id, []).copy()

    def list(
        self,
        config: Dict[str, Any],
        search: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[Message]:
        thread_id = config.get("thread_id", "default")
        messages = self._messages.get(thread_id, [])
        if search:
            messages = [m for m in messages if search in m.content]
        if offset is not None:
            messages = messages[offset:]
        if limit is not None:
            messages = messages[:limit]
        return messages.copy()

    def delete(self, config: Dict[str, Any]) -> None:
        thread_id = config.get("thread_id", "default")
        self._messages.pop(thread_id, None)

    def get_state(self, config: Dict[str, Any]) -> Optional[AgentState]:
        thread_id = config.get("thread_id", "default")
        return self._states.get(thread_id)

    def update_state(
        self,
        config: Dict[str, Any],
        state: AgentState,
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._states[thread_id] = state

    def put_thread(
        self,
        config: Dict[str, Any],
        thread_info: Dict[str, Any],
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._threads[thread_id] = thread_info.copy()

    def get_thread(
        self,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        thread_id = config.get("thread_id", "default")
        return self._threads.get(thread_id)

    def list_threads(
        self,
        search: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        threads = list(self._threads.values())
        if search:
            threads = [t for t in threads if search in str(t)]
        if offset is not None:
            threads = threads[offset:]
        if limit is not None:
            threads = threads[:limit]
        return [t.copy() for t in threads]

    def cleanup(
        self,
        config: Dict[str, Any],
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._messages.pop(thread_id, None)
        self._states.pop(thread_id, None)
        self._threads.pop(thread_id, None)
