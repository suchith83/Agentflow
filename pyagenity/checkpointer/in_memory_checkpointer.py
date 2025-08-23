import logging
from typing import Any, TypeVar

from pyagenity.state import AgentState
from pyagenity.utils import Message

from .base_checkpointer import BaseCheckpointer


# Define the type variable for this implementation
StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger(__name__)


class InMemoryCheckpointer(BaseCheckpointer[StateT]):
    """
    In-memory checkpointer that persists combined AgentState (including execution metadata).

    This class provides an in-memory implementation of the checkpointer interface, storing agent states,
    messages, threads, and sync states in Python dictionaries. It is suitable for testing, prototyping,
    or scenarios where persistence across process restarts is not required.

    Example:
        >>> from pyagenity.checkpointer.in_memory_checkpointer import InMemoryCheckpointer
        >>> from pyagenity.state import AgentState
        >>> checkpointer = InMemoryCheckpointer()
        >>> config = {"agent_id": "123"}
        >>> state = AgentState()  # Replace with actual state object
        >>> checkpointer.put_state(config, state)
        >>> retrieved = checkpointer.get_state(config)
        >>> assert retrieved == state
        >>> checkpointer.clear_state(config)
        >>> assert checkpointer.get_state(config) is None
    """

    def __init__(self):
        """
        Initializes the in-memory checkpointer with empty stores for states, messages, threads, and sync states.
        """
        logger.debug("Initializing InMemoryCheckpointer")
        self._state_store: dict[str, StateT] = {}
        self._messages_store: dict[str, list[Message]] = {}
        self._threads_store: dict[str, dict[str, Any]] = {}
        self._sync_state_store: dict[str, StateT] = {}

    def _config_key(self, config: dict[str, Any]) -> str:
        """
        Generates a unique string key from the configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary.

        Returns:
            str: A string key representing the configuration.
        """
        key = str(sorted(config.items()))
        logger.debug("Generated config key: %s", key[:50] + "..." if len(key) > 50 else key)
        return key

    # === PRIMARY API: Combined State Management ===
    def put_state(self, config: dict[str, Any], state: StateT) -> None:
        """
        Stores the agent state for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.
            state (StateT): The agent state to store.
        """
        key = self._config_key(config)
        self._state_store[key] = state
        logger.debug("Stored state for key: %s", key[:30] + "..." if len(key) > 30 else key)

    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieves the agent state for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.

        Returns:
            StateT | None: The stored agent state, or None if not found.
        """
        key = self._config_key(config)
        state = self._state_store.get(key)
        logger.debug(
            "Retrieved state for key: %s (found: %s)",
            key[:30] + "..." if len(key) > 30 else key,
            state is not None,
        )
        return state

    def clear_state(self, config: dict[str, Any]) -> None:
        """
        Removes the agent state for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.
        """
        key = self._config_key(config)
        removed = self._state_store.pop(key, None)
        logger.debug(
            "Cleared state for key: %s (was present: %s)",
            key[:30] + "..." if len(key) > 30 else key,
            removed is not None,
        )

    # === Realtime Sync State ===
    def sync_state(self, config: dict[str, Any], state: StateT) -> None:
        """
        Stores the real-time sync state for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.
            state (StateT): The sync state to store.
        """
        key = self._config_key(config)
        self._sync_state_store[key] = state
        logger.debug("Synced state for key: %s", key[:30] + "...")

    def get_sync_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieves the real-time sync state for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.

        Returns:
            StateT | None: The stored sync state, or None if not found.
        """
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
        """
        Stores a list of messages for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict[str, Any] | None): Optional metadata (unused).
        """
        key = self._config_key(config)
        self._messages_store[key] = messages
        logger.debug("Stored messages for key: %s", key[:30] + "...")

    def get_message(self, config: dict[str, Any]) -> Message:
        """
        Retrieves the most recent message for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.

        Returns:
            Message: The most recent message, or an empty list if none exist.
        """
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
        """
        Lists messages for the given configuration, with optional filtering and pagination.

        Args:
            config (dict[str, Any]): Configuration dictionary.
            search (str | None): Optional search string to filter messages.
            offset (int | None): Optional offset for pagination.
            limit (int | None): Optional limit for pagination.

        Returns:
            list[Message]: List of messages matching the criteria.
        """
        key = self._config_key(config)
        msgs = self._messages_store.get(key, [])
        if search:
            msgs = [m for m in msgs if search in str(m)]
        if offset is not None:
            msgs = msgs[offset:]
        if limit is not None:
            msgs = msgs[:limit]

        logger.debug("Listing all messages for key: %s", key[:30] + "...")
        return msgs

    def delete_message(self, config: dict[str, Any]) -> None:
        """
        Deletes all messages for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.
        """
        key = self._config_key(config)
        self._messages_store.pop(key, None)
        logger.debug("Deleted messages for key: %s", key[:30] + "...")

    def put_thread(self, config: dict[str, Any], thread_info: dict[str, Any]) -> None:
        """
        Stores thread information for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.
            thread_info (dict[str, Any]): Thread information to store.
        """
        key = self._config_key(config)
        self._threads_store[key] = thread_info
        logger.debug("Stored thread for key: %s", key[:30] + "...")

    def get_thread(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """
        Retrieves thread information for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.

        Returns:
            dict[str, Any] | None: The stored thread information, or None if not found.
        """
        key = self._config_key(config)
        logger.debug("Retrieved thread for key: %s", key[:30] + "...")
        return self._threads_store.get(key)

    def list_threads(
        self,
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Lists all stored threads, with optional filtering and pagination.

        Args:
            search (str | None): Optional search string to filter threads.
            offset (int | None): Optional offset for pagination.
            limit (int | None): Optional limit for pagination.

        Returns:
            list[dict[str, Any]]: List of thread information dictionaries.
        """
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
        """
        Removes all stored data (state, messages, threads, sync state) for the given configuration.

        Args:
            config (dict[str, Any]): Configuration dictionary.
        """
        key = self._config_key(config)
        self._state_store.pop(key, None)
        self._messages_store.pop(key, None)
        self._threads_store.pop(key, None)
        self._sync_state_store.pop(key, None)
        logger.debug("Cleaned up all data for key: %s", key[:30] + "...")
