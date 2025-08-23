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
    Base class for checkpointer implementations.

    Generic over state types to support custom AgentState subclasses.

    Primary API (new combined state approach):
    - put_state: Store complete AgentState (including execution metadata)
    - get_state: Retrieve complete AgentState
    - clear_state: Remove stored state
    """

    # Realtime Sync of state Recommended to use faster database
    def sync_state(self, config: dict[str, Any], state: StateT) -> None:
        """Sync the current state to a faster database for real-time access."""
        raise NotImplementedError("sync_state method must be implemented")

    def get_sync_state(self, config: dict[str, Any]) -> StateT | None:
        """Get the synced state from the faster database."""
        raise NotImplementedError("get_sync_state method must be implemented")

    # === PRIMARY API: Combined State Management ===

    @abstractmethod
    def put_state(
        self,
        config: dict[str, Any],
        state: StateT,
    ) -> None:
        """Store complete AgentState (including execution metadata) atomically.

        This is the primary method for persisting state in the new design.
        State includes both user data and internal execution metadata.
        """
        raise NotImplementedError("put_state method must be implemented")

    @abstractmethod
    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """Get the complete AgentState (including execution metadata).

        This is the primary method for retrieving state in the new design.
        Returns None if no state exists for the given config.
        """
        raise NotImplementedError("get_state method must be implemented")

    @abstractmethod
    def clear_state(
        self,
        config: dict[str, Any],
    ) -> None:
        """Clear the complete AgentState for the given config.

        This is the primary method for cleaning up state in the new design.
        """
        raise NotImplementedError("clear_state method must be implemented")

    # === OTHER METHODS: Messages, Threads, etc. ===

    @abstractmethod
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a checkpoint."""
        raise NotImplementedError("put method must be implemented")

    @abstractmethod
    def get_message(self, config: dict[str, Any]) -> Message:
        """Retrieve a checkpoint."""
        raise NotImplementedError("get method must be implemented")

    @abstractmethod
    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """List checkpoints for a thread."""
        raise NotImplementedError("list method must be implemented")

    def delete_message(self, config: dict[str, Any]) -> None:
        """Delete a checkpoint."""
        raise NotImplementedError("delete method must be implemented")

    def put_thread(
        self,
        config: dict[str, Any],
        thread_info: dict[str, Any],
    ) -> None:
        """Store a new thread."""
        raise NotImplementedError("put_thread method must be implemented")

    def get_thread(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Retrieve a thread by its ID."""
        raise NotImplementedError("get_thread method must be implemented")

    def list_threads(
        self,
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Lists all threads with optional search and pagination parameters.

        Args:
            search (str, optional): A search string to filter threads by name or content.
            offset (int, optional): The number of threads to skip before starting to collect the result set.
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
        Cleans up resources associated with the given configuration.
        This method should be implemented to delete all checkpoints related to a specific thread or configuration.
        It is intended to free up any resources or storage used by the checkpointing process.

        Args:
            config (dict[str, Any]): The configuration dictionary containing information about the resources to clean up.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("cleanup method must be implemented")
