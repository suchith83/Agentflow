import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from agentflow.state import AgentState, Message
from agentflow.utils import run_coroutine
from agentflow.utils.thread_info import ThreadInfo


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agentflow.state import AgentState, Message


StateT = TypeVar("StateT", bound="AgentState")


class BaseCheckpointer[StateT: AgentState](ABC):
    """
    Abstract base class for checkpointing agent state, messages, and threads.

    This class defines the contract for all checkpointer implementations, supporting both
    async and sync methods.
    Subclasses should implement async methods for optimal performance.
    Sync methods are provided for compatibility.

    Usage:
        - Async-first design: subclasses should implement `async def` methods.
        - If a subclass provides only a sync `def`, it will be executed in a worker thread
            automatically using `asyncio.run`.
        - Callers always use the async APIs (`await cp.put_state(...)`, etc.).

    Type Args:
        StateT: Type of agent state (must inherit from AgentState).
    """

    ###########################
    #### SETUP ################
    ###########################
    def setup(self) -> Any:
        """
        Synchronous setup method for checkpointer.

        Returns:
            Any: Implementation-defined setup result.
        """
        return run_coroutine(self.asetup())

    @abstractmethod
    async def asetup(self) -> Any:
        """
        Asynchronous setup method for checkpointer.

        Returns:
            Any: Implementation-defined setup result.
        """
        raise NotImplementedError

    # -------------------------
    # State methods Async
    # -------------------------
    @abstractmethod
    async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """
        Store agent state asynchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.
        """
        raise NotImplementedError

    @abstractmethod
    async def aclear_state(self, config: dict[str, Any]) -> Any:
        """
        Clear agent state asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any: Implementation-defined result.
        """
        raise NotImplementedError

    @abstractmethod
    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        """
        Store agent state in cache asynchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to cache.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state from cache asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Cached state or None.
        """
        raise NotImplementedError

    # -------------------------
    # State methods Sync
    # -------------------------
    def put_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """
        Store agent state synchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to store.

        Returns:
            StateT: The stored state object.
        """
        return run_coroutine(self.aput_state(config, state))

    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Retrieved state or None.
        """
        return run_coroutine(self.aget_state(config))

    def clear_state(self, config: dict[str, Any]) -> Any:
        """
        Clear agent state synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any: Implementation-defined result.
        """
        return run_coroutine(self.aclear_state(config))

    def put_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        """
        Store agent state in cache synchronously.

        Args:
            config (dict): Configuration dictionary.
            state (StateT): State object to cache.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.aput_state_cache(config, state))

    def get_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve agent state from cache synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            StateT | None: Cached state or None.
        """
        return run_coroutine(self.aget_state_cache(config))

    # -------------------------
    # Message methods async
    # -------------------------
    @abstractmethod
    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Store messages asynchronously.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Additional metadata.

        Returns:
            Any: Implementation-defined result.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """
        Retrieve a specific message asynchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Message: Retrieved message object.
        """
        raise NotImplementedError

    @abstractmethod
    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        List messages asynchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[Message]: List of message objects.
        """
        raise NotImplementedError

    @abstractmethod
    async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> Any | None:
        """
        Delete a specific message asynchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    # -------------------------
    # Message methods sync
    # -------------------------
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Store messages synchronously.

        Args:
            config (dict): Configuration dictionary.
            messages (list[Message]): List of messages to store.
            metadata (dict, optional): Additional metadata.

        Returns:
            Any: Implementation-defined result.
        """
        return run_coroutine(self.aput_messages(config, messages, metadata))

    def get_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """
        Retrieve a specific message synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Message: Retrieved message object.
        """
        return run_coroutine(self.aget_message(config, message_id))

    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        List messages synchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[Message]: List of message objects.
        """
        return run_coroutine(self.alist_messages(config, search, offset, limit))

    def delete_message(self, config: dict[str, Any], message_id: str | int) -> Any | None:
        """
        Delete a specific message synchronously.

        Args:
            config (dict): Configuration dictionary.
            message_id (str|int): Message identifier.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.adelete_message(config, message_id))

    # -------------------------
    # Thread methods async
    # -------------------------
    @abstractmethod
    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> Any | None:
        """
        Store thread info asynchronously.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    @abstractmethod
    async def aget_thread(
        self,
        config: dict[str, Any],
    ) -> ThreadInfo | None:
        """
        Retrieve thread info asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ThreadInfo | None: Thread information object or None.
        """
        raise NotImplementedError

    @abstractmethod
    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """
        List threads asynchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[ThreadInfo]: List of thread information objects.
        """
        raise NotImplementedError

    @abstractmethod
    async def aclean_thread(self, config: dict[str, Any]) -> Any | None:
        """
        Clean/delete thread asynchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError

    # -------------------------
    # Thread methods sync
    # -------------------------
    def put_thread(self, config: dict[str, Any], thread_info: ThreadInfo) -> Any | None:
        """
        Store thread info synchronously.

        Args:
            config (dict): Configuration dictionary.
            thread_info (ThreadInfo): Thread information object.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.aput_thread(config, thread_info))

    def get_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        """
        Retrieve thread info synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            ThreadInfo | None: Thread information object or None.
        """
        return run_coroutine(self.aget_thread(config))

    def list_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """
        List threads synchronously with optional filtering.

        Args:
            config (dict): Configuration dictionary.
            search (str, optional): Search string.
            offset (int, optional): Offset for pagination.
            limit (int, optional): Limit for pagination.

        Returns:
            list[ThreadInfo]: List of thread information objects.
        """
        return run_coroutine(self.alist_threads(config, search, offset, limit))

    def clean_thread(self, config: dict[str, Any]) -> Any | None:
        """
        Clean/delete thread synchronously.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.aclean_thread(config))

    # -------------------------
    # Clean Resources
    # -------------------------
    def release(self) -> Any | None:
        """
        Release resources synchronously.

        Returns:
            Any | None: Implementation-defined result.
        """
        return run_coroutine(self.arelease())

    @abstractmethod
    async def arelease(self) -> Any | None:
        """
        Release resources asynchronously.

        Returns:
            Any | None: Implementation-defined result.
        """
        raise NotImplementedError
