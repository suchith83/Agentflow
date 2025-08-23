import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import TypeVar

from .agent_state import AgentState


S = TypeVar("S", bound=AgentState)

logger = logging.getLogger(__name__)


class BaseContextManager[S](ABC):
    """
    Abstract base class for context management in AI interactions.

    Subclasses should implement `trim_context` as either a synchronous or asynchronous method.
    Generic over AgentState or its subclasses.
    """

    @abstractmethod
    def trim_context(self, state: S) -> S | Awaitable[S]:
        """
        Trim context based on message count. Can be sync or async.

        Subclasses may implement as either a synchronous or asynchronous method.

        Args:
            state: The state containing context to be trimmed.

        Returns:
            The state with trimmed context, either directly or as an awaitable.
        """
        raise NotImplementedError("Subclasses must implement this method (sync or async)")
