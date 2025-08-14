from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStore(ABC):
    """Base class for message storage implementations."""

    @abstractmethod
    def update_memory(
        self,
        config: Dict[str, Any],
        info: str,
    ) -> None:
        """Store a single message."""
        raise NotImplementedError("update_memory method must be implemented")

    def get_memory(
        self,
        config: Dict[str, Any],
    ) -> None:
        """Retrieve a single message."""
        raise NotImplementedError("get_memory method must be implemented")

    def delete_memory(
        self,
        config: Dict[str, Any],
    ) -> None:
        """Delete a single message."""
        raise NotImplementedError("delete_memory method must be implemented")

    def related_memory(
        self,
        config: Dict[str, Any],
        query: str,
    ) -> None:
        """Retrieve related messages."""
        raise NotImplementedError("related_memory method must be implemented")
