from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pyagenity.graph.state.state import AgentState
from pyagenity.graph.utils.message import Message


class BaseCheckpointer(ABC):
    """Base class for checkpointer implementations."""

    # CURD Operations for messages
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

    def delete_thread(self, config: Dict[str, Any]) -> None:
        """Delete all checkpoints for a thread."""
        raise NotImplementedError("delete_thread method must be implemented")

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
