from abc import ABC, abstractmethod
from typing import Any

from .events import EventModel


class BasePublisher(ABC):
    """Abstract base class for event publishers.

    This class defines the interface for publishing events. Subclasses should implement
    the publish, close, and sync_close methods to provide specific publishing logic.

    Attributes:
        config: Configuration dictionary for the publisher.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the publisher with the given configuration.

        Args:
            config: Configuration dictionary for the publisher.
        """
        self.config = config

    @abstractmethod
    async def publish(self, event: EventModel) -> Any:
        """Publish an event.

        Args:
            event: The event to publish.

        Returns:
            The result of the publish operation.
        """
        raise NotImplementedError

    @abstractmethod
    async def close(self):
        """Close the publisher and release any resources.

        This method should be overridden by subclasses to provide specific cleanup logic.
        It will be called externally.
        """
        raise NotImplementedError

    @abstractmethod
    def sync_close(self):
        """Close the publisher and release any resources (synchronous version).

        This method should be overridden by subclasses to provide specific cleanup logic.
        It will be called externally.
        """
        raise NotImplementedError
