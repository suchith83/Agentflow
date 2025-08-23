import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from pyagenity.state import AgentState
from pyagenity.utils import Message


# Generic type variable bound to AgentState for checkpointer subtyping
StateT = TypeVar("StateT", bound="AgentState")

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    pass


class BaseCheckpointer[StateT: AgentState](ABC):
    """
    Abstract base class for implementing checkpointing mechanisms for agent state management.

    This class provides a generic interface for storing, retrieving, and managing agent state,
    messages, and threads. It is designed to be subclassed for specific storage backends
    (e.g., databases, filesystems, cloud storage).

    The class is generic over state types to support custom `AgentState` subclasses.

    Example:
        ```python
        from pyagenity.checkpointer.base_checkpointer import BaseCheckpointer
        from pyagenity.state import AgentState


        class MyCheckpointer(BaseCheckpointer[AgentState]):
            def put_state(self, config, state):
                # Store state in your backend
                pass

            def get_state(self, config):
                # Retrieve state from your backend
                pass

            def clear_state(self, config):
                # Remove state from your backend
                pass

            # Implement other required methods...


        # Usage
        config = {"thread_id": "abc123"}
        state = AgentState(...)
        checkpointer = MyCheckpointer()
        checkpointer.put_state(config, state)
        restored_state = checkpointer.get_state(config)
        ```

    Attributes:
        None (all methods are intended to be implemented by subclasses).
    """

    def sync_state(self, config: dict[str, Any], state: StateT) -> None:
        """
        Sync the current state to a faster database for real-time access.

        This method is recommended for use with high-performance databases (e.g., Redis)
        to enable real-time state synchronization.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the state context.
            state (StateT): The agent state to be synced.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("sync_state method must be implemented")

    def get_sync_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve the synced state from the faster database.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the state context.

        Returns:
            StateT | None: The synced agent state, or None if not found.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("get_sync_state method must be implemented")

    @abstractmethod
    def put_state(
        self,
        config: dict[str, Any],
        state: StateT,
    ) -> None:
        """
        Store the complete AgentState (including execution metadata) atomically.

        This is the primary method for persisting state in the new design.
        State includes both user data and internal execution metadata.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the state context.
            state (StateT): The agent state to be stored.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("put_state method must be implemented")

    @abstractmethod
    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """
        Retrieve the complete AgentState (including execution metadata).

        This is the primary method for retrieving state in the new design.
        Returns None if no state exists for the given config.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the state context.

        Returns:
            StateT | None: The retrieved agent state, or None if not found.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("get_state method must be implemented")

    @abstractmethod
    def clear_state(
        self,
        config: dict[str, Any],
    ) -> None:
        """
        Clear the complete AgentState for the given config.

        This is the primary method for cleaning up state in the new design.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the state context.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("clear_state method must be implemented")

    @abstractmethod
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a list of messages as a checkpoint.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the message context.
            messages (list[Message]): List of messages to store.
            metadata (dict[str, Any], optional): Additional metadata for the messages.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("put method must be implemented")

    @abstractmethod
    def get_message(self, config: dict[str, Any]) -> Message:
        """
        Retrieve a checkpointed message.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the message context.

        Returns:
            Message: The retrieved message.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("get method must be implemented")

    @abstractmethod
    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """
        List checkpoints for a thread.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the thread context.
            search (str, optional): Search string to filter messages.
            offset (int, optional): Number of messages to skip.
            limit (int, optional): Maximum number of messages to return.

        Returns:
            list[Message]: List of messages matching the criteria.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("list method must be implemented")

    def delete_message(self, config: dict[str, Any]) -> None:
        """
        Delete a checkpointed message.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the message context.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("delete method must be implemented")

    def put_thread(
        self,
        config: dict[str, Any],
        thread_info: dict[str, Any],
    ) -> None:
        """
        Store a new thread.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the thread context.
            thread_info (dict[str, Any]): Information about the thread to store.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("put_thread method must be implemented")

    def get_thread(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Retrieve a thread by its ID.

        Args:
            config (dict[str, Any]): Configuration dictionary identifying the thread context.

        Returns:
            dict[str, Any] | None: The thread information, or None if not found.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("get_thread method must be implemented")

    def list_threads(
        self,
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        List all threads with optional search and pagination parameters.

        Args:
            search (str, optional): A search string to filter threads by name or content.
            offset (int, optional): The number of threads to skip before starting to
                collect the result set.
            limit (int, optional): The maximum number of threads to return.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each representing a thread.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("list_threads method must be implemented")

    def cleanup(
        self,
        config: dict[str, Any],
    ) -> None:
        """
        Clean up resources associated with the given configuration.

        This method should be implemented to delete all checkpoints related to a specific thread
        or configuration. It is intended to free up any resources or storage used by the
        checkpointing process.

        Args:
            config (dict[str, Any]): The configuration dictionary containing information about
            the resources to clean up.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("cleanup method must be implemented")
