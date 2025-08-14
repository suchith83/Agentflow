from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pyagenity.graph.state.state import AgentState
from pyagenity.graph.utils.message import Message


class BaseCheckpointer(ABC):
    """
    Base class for checkpointer implementations.
    Will Create 3 Tables:
    1. Thread Table (Save thread information)
    2. Message Table (Save message information)
    3. State Table (Save state information)
    """

    @abstractmethod
    def put(
        self,
        config: Dict[str, Any],
        messages: list[Message],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a checkpoint."""
        raise NotImplementedError("put method must be implemented")

    @abstractmethod
    def get(self, config: Dict[str, Any]) -> list[Message]:
        """Retrieve a checkpoint."""
        raise NotImplementedError("get method must be implemented")

    @abstractmethod
    def list(
        self,
        config: Dict[str, Any],
        search: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[Message]:
        """List checkpoints for a thread."""
        raise NotImplementedError("list method must be implemented")

    def delete(self, config: Dict[str, Any]) -> None:
        """Delete a checkpoint."""
        raise NotImplementedError("delete method must be implemented")

    def get_state(self, config: Dict[str, Any]) -> Optional[AgentState]:
        """Get the latest state snapshot."""
        raise NotImplementedError("get_state method must be implemented")

    def update_state(
        self,
        config: Dict[str, Any],
        state: AgentState,
    ) -> None:
        """Update the state at the current checkpoint."""
        raise NotImplementedError("update_state method must be implemented")

    def put_thread(
        self,
        config: Dict[str, Any],
        thread_info: Dict[str, Any],
    ) -> None:
        """Store a new thread."""
        raise NotImplementedError("put_thread method must be implemented")

    def get_thread(
        self,
        config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a thread by its ID."""
        raise NotImplementedError("get_thread method must be implemented")

    def list_threads(
        self,
        search: Optional[str] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List all threads."""
        raise NotImplementedError("list_threads method must be implemented")

    def cleanup(
        self,
        config: Dict[str, Any],
    ) -> None:
        """Cleanup resources if needed, This will delete all checkpoints for a thread."""
        pass
