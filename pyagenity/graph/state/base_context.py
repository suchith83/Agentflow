from abc import ABC, abstractmethod

from pyagenity.graph.state import AgentState


class BaseContextManager(ABC):
    """Abstract base class for context management in AI interactions."""

    @abstractmethod
    def trim_context(self, state: AgentState) -> AgentState:
        """Trim context based on message count."""
        raise NotImplementedError("Subclasses must implement this method")

    async def atrim_context(self, state: AgentState) -> AgentState:
        """Asynchronous method to trim context based on message count."""
        raise NotImplementedError("Subclasses must implement this method")
