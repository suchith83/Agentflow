from typing import Any, TypeVar

from pyagenity.graph.state import AgentState
from pyagenity.graph.state.execution_state import ExecutionState
from pyagenity.graph.utils import Message

from .base_checkpointer import BaseCheckpointer

# Define the type variable for this implementation
StateT = TypeVar("StateT", bound=AgentState)


class InMemoryCheckpointer(BaseCheckpointer[AgentState]):
    """In-memory checkpointer that persists combined AgentState (including execution metadata)."""

    def __init__(self):
        # Use dicts for in-memory storage
        self._state_store: dict[str, AgentState] = {}
        self._messages_store: dict[str, list[Message]] = {}
        self._threads_store: dict[str, dict[str, Any]] = {}
        self._sync_state_store: dict[str, AgentState] = {}

    def _config_key(self, config: dict[str, Any]) -> str:
        # Use a string key for config; simple str() for demo, customize as needed
        return str(sorted(config.items()))

    # === PRIMARY API: Combined State Management ===
    def put_state(self, config: dict[str, Any], state: AgentState) -> None:
        key = self._config_key(config)
        self._state_store[key] = state

    def get_state(self, config: dict[str, Any]) -> AgentState | None:
        key = self._config_key(config)
        return self._state_store.get(key)

    def clear_state(self, config: dict[str, Any]) -> None:
        key = self._config_key(config)
        self._state_store.pop(key, None)

    # === Realtime Sync State ===
    def sync_state(self, config: dict[str, Any], state: AgentState) -> None:
        key = self._config_key(config)
        self._sync_state_store[key] = state

    def get_sync_state(self, config: dict[str, Any]) -> AgentState | None:
        key = self._config_key(config)
        return self._sync_state_store.get(key)

    # === OTHER METHODS: Messages, Threads, etc. ===
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        key = self._config_key(config)
        self._messages_store[key] = messages

    def get_message(self, config: dict[str, Any]) -> Message:
        key = self._config_key(config)
        msgs = self._messages_store.get(key, [])
        if not msgs:
            raise KeyError(f"No messages for config: {config}")
        return msgs[-1]

    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        key = self._config_key(config)
        msgs = self._messages_store.get(key, [])
        # Optionally filter/search
        if search:
            msgs = [m for m in msgs if search in str(m)]
        if offset is not None:
            msgs = msgs[offset:]
        if limit is not None:
            msgs = msgs[:limit]
        return msgs

    def delete_message(self, config: dict[str, Any]) -> None:
        key = self._config_key(config)
        self._messages_store.pop(key, None)

    def put_thread(self, config: dict[str, Any], thread_info: dict[str, Any]) -> None:
        key = self._config_key(config)
        self._threads_store[key] = thread_info

    def get_thread(self, config: dict[str, Any]) -> dict[str, Any] | None:
        key = self._config_key(config)
        return self._threads_store.get(key)

    def list_threads(
        self,
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        threads = list(self._threads_store.values())
        if search:
            threads = [t for t in threads if search in str(t)]
        if offset is not None:
            threads = threads[offset:]
        if limit is not None:
            threads = threads[:limit]
        return threads

    def cleanup(self, config: dict[str, Any]) -> None:
        key = self._config_key(config)
        self._state_store.pop(key, None)
        self._messages_store.pop(key, None)
        self._threads_store.pop(key, None)
        self._sync_state_store.pop(key, None)
