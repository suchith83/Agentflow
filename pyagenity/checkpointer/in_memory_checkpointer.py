import logging
from typing import Any, TypeVar

from pyagenity.state import AgentState
from pyagenity.utils import Message

from .base_checkpointer import BaseCheckpointer


# Define the type variable for this implementation
StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger(__name__)


class InMemoryCheckpointer(BaseCheckpointer[StateT]):
    """In-memory checkpointer that persists combined AgentState (including execution metadata)."""

    def __init__(self):
        logger.debug("Initializing InMemoryCheckpointer")
        # Use dicts for in-memory storage
        self._state_store: dict[str, StateT] = {}
        self._messages_store: dict[str, list[Message]] = {}
        self._threads_store: dict[str, dict[str, Any]] = {}
        self._sync_state_store: dict[str, StateT] = {}

    def _config_key(self, config: dict[str, Any]) -> str:
        # Use a string key for config; simple str() for demo, customize as needed
        key = str(sorted(config.items()))
        logger.debug("Generated config key: %s", key[:50] + "..." if len(key) > 50 else key)
        return key

    # === PRIMARY API: Combined State Management ===
    def put_state(self, config: dict[str, Any], state: StateT) -> None:
        key = self._config_key(config)
        self._state_store[key] = state
        logger.debug("Stored state for key: %s", key[:30] + "..." if len(key) > 30 else key)

    def get_state(self, config: dict[str, Any]) -> StateT | None:
        key = self._config_key(config)
        state = self._state_store.get(key)
        logger.debug(
            "Retrieved state for key: %s (found: %s)",
            key[:30] + "..." if len(key) > 30 else key,
            state is not None,
        )
        return state

    def clear_state(self, config: dict[str, Any]) -> None:
        key = self._config_key(config)
        removed = self._state_store.pop(key, None)
        logger.debug(
            "Cleared state for key: %s (was present: %s)",
            key[:30] + "..." if len(key) > 30 else key,
            removed is not None,
        )

    # === Realtime Sync State ===
    def sync_state(self, config: dict[str, Any], state: StateT) -> None:
        key = self._config_key(config)
        self._sync_state_store[key] = state
        logger.debug("Synced state for key: %s", key[:30] + "...")

    def get_sync_state(self, config: dict[str, Any]) -> StateT | None:
        key = self._config_key(config)
        logger.debug("Retrieved sync state for key: %s", key[:30] + "...")
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
        logger.debug("Stored messages for key: %s", key[:30] + "...")

    def get_message(self, config: dict[str, Any]) -> Message:
        key = self._config_key(config)
        msgs = self._messages_store.get(key, [])
        logger.debug("Retrieved message for key: %s (found: %s)", key[:30] + "...", bool(msgs))
        return msgs[-1] if msgs else []  # type: ignore

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

        logger.debug("Listing all messages for key: %s", key[:30] + "...")
        return msgs

    def delete_message(self, config: dict[str, Any]) -> None:
        key = self._config_key(config)
        self._messages_store.pop(key, None)
        logger.debug("Deleted messages for key: %s", key[:30] + "...")

    def put_thread(self, config: dict[str, Any], thread_info: dict[str, Any]) -> None:
        key = self._config_key(config)
        self._threads_store[key] = thread_info
        logger.debug("Stored thread for key: %s", key[:30] + "...")

    def get_thread(self, config: dict[str, Any]) -> dict[str, Any] | None:
        key = self._config_key(config)
        logger.debug("Retrieved thread for key: %s", key[:30] + "...")
        return self._threads_store.get(key)

    def list_threads(
        self,
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        threads = list(self._threads_store.values())
        logger.debug("Listing all threads: %s", threads)
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
        logger.debug("Cleaned up all data for key: %s", key[:30] + "...")
