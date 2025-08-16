from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar


# Generic type variable for extensible data types
DataT = TypeVar("DataT")


class BaseStore(Generic[DataT], ABC):
    """Base class for message storage implementations.

    Generic over data types to support extensible storage formats.
    """

    @abstractmethod
    def update_memory(
        self,
        config: dict[str, Any],
        info: DataT,
    ) -> None:
        """Store a single piece of information."""
        raise NotImplementedError("update_memory method must be implemented")

    def get_memory(
        self,
        config: dict[str, Any],
    ) -> DataT | None:
        """Retrieve a single piece of information."""
        raise NotImplementedError("get_memory method must be implemented")

    def delete_memory(
        self,
        config: dict[str, Any],
    ) -> None:
        """Delete a single piece of information."""
        raise NotImplementedError("delete_memory method must be implemented")

    def related_memory(
        self,
        config: dict[str, Any],
        query: str,
    ) -> list[DataT]:
        """Retrieve related information."""
        raise NotImplementedError("related_memory method must be implemented")
