from typing import Any

from pyagenity.graph.state import AgentState
from pyagenity.graph.utils import Message

from .base_checkpointer import BaseCheckpointer


class InMemoryCheckpointer(BaseCheckpointer):
    def __init__(self):
        # Simulate tables with dicts/lists
        self._threads: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[Message]] = {}
        self._states: dict[str, AgentState] = {}

    def put(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._messages[thread_id] = messages.copy()
        if metadata:
            self._threads.setdefault(thread_id, {})["metadata"] = metadata

    def get(self, config: dict[str, Any]) -> list[Message]:
        thread_id = config.get("thread_id", "default")
        return self._messages.get(thread_id, []).copy()

    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
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

    def delete(self, config: dict[str, Any]) -> None:
        thread_id = config.get("thread_id", "default")
        self._messages.pop(thread_id, None)

    def get_state(self, config: dict[str, Any]) -> AgentState | None:
        thread_id = config.get("thread_id", "default")
        return self._states.get(thread_id)

    def update_state(
        self,
        config: dict[str, Any],
        state: AgentState,
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._states[thread_id] = state

    def put_thread(
        self,
        config: dict[str, Any],
        thread_info: dict[str, Any],
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._threads[thread_id] = thread_info.copy()

    def get_thread(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        thread_id = config.get("thread_id", "default")
        return self._threads.get(thread_id)

    def list_threads(
        self,
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
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
        config: dict[str, Any],
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._messages.pop(thread_id, None)
        self._states.pop(thread_id, None)
        self._threads.pop(thread_id, None)
