from abc import ABC, abstractmethod
from collections.abc import Awaitable

from pyagenity.graph.state import AgentState


class BaseContextManager[S: (AgentState)](ABC):
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
        """
        raise NotImplementedError("Subclasses must implement this method (sync or async)")
