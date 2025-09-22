import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, TypeVar

from pyagenity.utils import Message
from pyagenity.utils.thread_info import ThreadInfo

from .base_checkpointer import BaseCheckpointer


if TYPE_CHECKING:
    from pyagenity.state import AgentState

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound="AgentState")


class InMemoryCheckpointer[StateT: AgentState](BaseCheckpointer[StateT]):
    """
    In-memory implementation of BaseCheckpointer.

    Stores all data in memory using dictionaries. Data is lost when the process ends.
    Async-first design using asyncio locks for concurrent access.
    """

    def __init__(self):
        # State storage
        self._states: dict[str, StateT] = {}
        self._state_cache: dict[str, StateT] = {}

        # Message storage - organized by config key
        self._messages: dict[str, list[Message]] = defaultdict(list)
        self._message_metadata: dict[str, dict[str, Any]] = {}

        # Thread storage
        self._threads: dict[str, dict[str, Any]] = {}

        # Async locks for concurrent access
        self._state_lock = asyncio.Lock()
        self._messages_lock = asyncio.Lock()
        self._threads_lock = asyncio.Lock()

    def setup(self) -> Any:
        logger.debug("InMemoryCheckpointer setup not required")

    async def asetup(self) -> Any:
        logger.debug("InMemoryCheckpointer async setup not required")

    def _get_config_key(self, config: dict[str, Any]) -> str:
        """Generate a string key from config dict for storage indexing."""
        # Sort keys for consistent hashing
        thread_id = config.get("thread_id", "")
        return str(thread_id)

    # -------------------------
    # State methods Async
    # -------------------------
    async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """Store state asynchronously."""
        key = self._get_config_key(config)
        async with self._state_lock:
            self._states[key] = state
            logger.debug(f"Stored state for key: {key}")
            return state

    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        """Retrieve state asynchronously."""
        key = self._get_config_key(config)
        async with self._state_lock:
            state = self._states.get(key)
            logger.debug(f"Retrieved state for key: {key}, found: {state is not None}")
            return state

    async def aclear_state(self, config: dict[str, Any]) -> bool:
        """Clear state asynchronously."""
        key = self._get_config_key(config)
        async with self._state_lock:
            if key in self._states:
                del self._states[key]
                logger.debug(f"Cleared state for key: {key}")
            return True

    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> StateT:
        """Store state cache asynchronously."""
        key = self._get_config_key(config)
        async with self._state_lock:
            self._state_cache[key] = state
            logger.debug(f"Stored state cache for key: {key}")
            return state

    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """Retrieve state cache asynchronously."""
        key = self._get_config_key(config)
        async with self._state_lock:
            cache = self._state_cache.get(key)
            logger.debug(f"Retrieved state cache for key: {key}, found: {cache is not None}")
            return cache

    # -------------------------
    # State methods Sync
    # -------------------------
    def put_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """Store state synchronously."""
        key = self._get_config_key(config)
        # For sync methods, we'll use a simple approach without locks
        # In a real async-first system, sync methods might not be used
        self._states[key] = state
        logger.debug(f"Stored state for key: {key}")
        return state

    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """Retrieve state synchronously."""
        key = self._get_config_key(config)
        state = self._states.get(key)
        logger.debug(f"Retrieved state for key: {key}, found: {state is not None}")
        return state

    def clear_state(self, config: dict[str, Any]) -> bool:
        """Clear state synchronously."""
        key = self._get_config_key(config)
        if key in self._states:
            del self._states[key]
            logger.debug(f"Cleared state for key: {key}")
        return True

    def put_state_cache(self, config: dict[str, Any], state: StateT) -> StateT:
        """Store state cache synchronously."""
        key = self._get_config_key(config)
        self._state_cache[key] = state
        logger.debug(f"Stored state cache for key: {key}")
        return state

    def get_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """Retrieve state cache synchronously."""
        key = self._get_config_key(config)
        cache = self._state_cache.get(key)
        logger.debug(f"Retrieved state cache for key: {key}, found: {cache is not None}")
        return cache

    # -------------------------
    # Message methods async
    # -------------------------
    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store messages asynchronously."""
        key = self._get_config_key(config)
        async with self._messages_lock:
            self._messages[key].extend(messages)
            if metadata:
                self._message_metadata[key] = metadata
            logger.debug(f"Stored {len(messages)} messages for key: {key}")
            return True

    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """Retrieve a specific message asynchronously."""
        key = self._get_config_key(config)
        async with self._messages_lock:
            messages = self._messages.get(key, [])
            for msg in messages:
                if msg.message_id == message_id:
                    return msg
            raise IndexError(f"Message with ID {message_id} not found for config key: {key}")

    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """List messages asynchronously with optional filtering."""
        key = self._get_config_key(config)
        async with self._messages_lock:
            messages = self._messages.get(key, [])

            # Apply search filter if provided
            if search:
                # Simple string search in message content
                messages = [
                    msg
                    for msg in messages
                    if hasattr(msg, "content") and search.lower() in str(msg.content).lower()
                ]

            # Apply offset and limit
            start = offset or 0
            end = (start + limit) if limit else None
            return messages[start:end]

    async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> bool:
        """Delete a specific message asynchronously."""
        key = self._get_config_key(config)
        async with self._messages_lock:
            messages = self._messages.get(key, [])
            for msg in messages:
                if msg.message_id == message_id:
                    messages.remove(msg)
                    logger.debug(f"Deleted message with ID {message_id} for key: {key}")
                    return True
            raise IndexError(f"Message with ID {message_id} not found for config key: {key}")

    # -------------------------
    # Message methods sync
    # -------------------------
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store messages synchronously."""
        key = self._get_config_key(config)
        self._messages[key].extend(messages)
        if metadata:
            self._message_metadata[key] = metadata

        logger.debug(f"Stored {len(messages)} messages for key: {key}")
        return True

    def get_message(self, config: dict[str, Any]) -> Message:
        """Retrieve the latest message synchronously."""
        key = self._get_config_key(config)
        messages = self._messages.get(key, [])
        if not messages:
            raise IndexError(f"No messages found for config key: {key}")
        return messages[-1]

    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """List messages synchronously with optional filtering."""
        key = self._get_config_key(config)
        messages = self._messages.get(key, [])

        # Apply search filter if provided
        if search:
            messages = [
                msg
                for msg in messages
                if hasattr(msg, "content") and search.lower() in str(msg.content).lower()
            ]

        # Apply offset and limit
        start = offset or 0
        end = (start + limit) if limit else None
        return messages[start:end]

    def delete_message(self, config: dict[str, Any], message_id: str | int) -> bool:
        """Delete a specific message synchronously."""
        key = self._get_config_key(config)
        messages = self._messages.get(key, [])
        for msg in messages:
            if msg.message_id == message_id:
                messages.remove(msg)
                logger.debug(f"Deleted message with ID {message_id} for key: {key}")
                return True
        raise IndexError(f"Message with ID {message_id} not found for config key: {key}")

    # -------------------------
    # Thread methods async
    # -------------------------
    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> bool:
        """Store thread info asynchronously."""
        key = self._get_config_key(config)
        async with self._threads_lock:
            self._threads[key] = thread_info.model_dump()
            logger.debug(f"Stored thread info for key: {key}")
            return True

    async def aget_thread(
        self,
        config: dict[str, Any],
    ) -> ThreadInfo | None:
        """Retrieve thread info asynchronously."""
        key = self._get_config_key(config)
        async with self._threads_lock:
            thread = self._threads.get(key)
            logger.debug(f"Retrieved thread for key: {key}, found: {thread is not None}")
            return ThreadInfo.model_validate(thread) if thread else None

    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """List all threads asynchronously with optional filtering."""
        async with self._threads_lock:
            threads = list(self._threads.values())

            # Apply search filter if provided
            if search:
                threads = [
                    thread
                    for thread in threads
                    if any(search.lower() in str(value).lower() for value in thread.values())
                ]

            # Apply offset and limit
            start = offset or 0
            end = (start + limit) if limit else None
            return [ThreadInfo.model_validate(thread) for thread in threads[start:end]]

    async def aclean_thread(self, config: dict[str, Any]) -> bool:
        """Clean/delete thread asynchronously."""
        key = self._get_config_key(config)
        async with self._threads_lock:
            if key in self._threads:
                del self._threads[key]
                logger.debug(f"Cleaned thread for key: {key}")
                return True
        return False

    # -------------------------
    # Thread methods sync
    # -------------------------
    def put_thread(self, config: dict[str, Any], thread_info: ThreadInfo) -> bool:
        """Store thread info synchronously."""
        key = self._get_config_key(config)
        self._threads[key] = thread_info.model_dump()
        logger.debug(f"Stored thread info for key: {key}")
        return True

    def get_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        """Retrieve thread info synchronously."""
        key = self._get_config_key(config)
        thread = self._threads.get(key)
        logger.debug(f"Retrieved thread for key: {key}, found: {thread is not None}")
        return ThreadInfo.model_validate(thread) if thread else None

    def list_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """List all threads synchronously with optional filtering."""
        threads = list(self._threads.values())

        # Apply search filter if provided
        if search:
            threads = [
                thread
                for thread in threads
                if any(search.lower() in str(value).lower() for value in thread.values())
            ]

        # Apply offset and limit
        start = offset or 0
        end = (start + limit) if limit else None
        return [ThreadInfo.model_validate(thread) for thread in threads[start:end]]

    def clean_thread(self, config: dict[str, Any]) -> bool:
        """Clean/delete thread synchronously."""
        key = self._get_config_key(config)
        if key in self._threads:
            del self._threads[key]
            logger.debug(f"Cleaned thread for key: {key}")
            return True
        return False

    # -------------------------
    # Clean Resources
    # -------------------------
    async def arelease(self) -> bool:
        """Release resources asynchronously."""
        async with self._state_lock, self._messages_lock, self._threads_lock:
            self._states.clear()
            self._state_cache.clear()
            self._messages.clear()
            self._message_metadata.clear()
            self._threads.clear()
            logger.info("Released all in-memory resources")
            return True

    def release(self) -> bool:
        """Release resources synchronously."""
        self._states.clear()
        self._state_cache.clear()
        self._messages.clear()
        self._message_metadata.clear()
        self._threads.clear()
        logger.info("Released all in-memory resources")
        return True
