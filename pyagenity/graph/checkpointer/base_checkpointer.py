from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pyagenity.graph.state import AgentState
from pyagenity.graph.state.execution_state import ExecutionState
from pyagenity.graph.utils import Message

if TYPE_CHECKING:
    pass


class BaseCheckpointer(ABC):
    """
    Base class for checkpointer implementations.

    Primary API (new combined state approach):
    - put_state: Store complete AgentState (including execution metadata)
    - get_state: Retrieve complete AgentState
    - clear_state: Remove stored state

    Legacy API (for backward compatibility):
    - put_execution_state/get_execution_state: Now implemented via combined state
    - Other methods remain for messages, threads, etc.
    """

    # === PRIMARY API: Combined State Management ===

    @abstractmethod
    def put_state(
        self,
        config: dict[str, Any],
        state: AgentState,
    ) -> None:
        """Store complete AgentState (including execution metadata) atomically.

        This is the primary method for persisting state in the new design.
        State includes both user data and internal execution metadata.
        """
        raise NotImplementedError("put_state method must be implemented")

    @abstractmethod
    def get_state(self, config: dict[str, Any]) -> AgentState | None:
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

    # === LEGACY API: For Backward Compatibility ===

    def update_state(
        self,
        config: dict[str, Any],
        state: AgentState,
    ) -> None:
        """Update the state at the current checkpoint.

        Legacy method - now delegates to put_state for compatibility.
        """
        self.put_state(config, state)

    def put_execution_state(
        self,
        config: dict[str, Any],
        execution_state: "ExecutionState",
    ) -> None:
        """Store execution state for pause/resume functionality.

        Legacy method - now implemented via combined state approach.
        This method gets or creates an AgentState and updates its execution_meta.
        """
        # Get existing state or create minimal one
        state = self.get_state(config)
        if state is None:
            # Create minimal AgentState with just execution metadata
            state = AgentState()
            state.execution_meta = execution_state
        else:
            # Update execution metadata in existing state
            state.execution_meta = execution_state

        # Store via primary API
        self.put_state(config, state)

    def get_execution_state(
        self,
        config: dict[str, Any],
    ) -> "ExecutionState | None":
        """Retrieve execution state for pause/resume functionality.

        Legacy method - now implemented via combined state approach.
        Extracts execution metadata from the stored AgentState.
        """
        state = self.get_state(config)
        if state is not None:
            return state.execution_meta
        return None

    def clear_execution_state(
        self,
        config: dict[str, Any],
    ) -> None:
        """Clear execution state when execution completes or errors.

        Legacy method - now implemented via combined state approach.
        Resets execution metadata in the AgentState but preserves other state.
        """
        state = self.get_state(config)
        if state is not None:
            # Reset execution metadata to initial state
            state.execution_meta = ExecutionState(current_node="__start__")
            self.put_state(config, state)

    # === OTHER METHODS: Messages, Threads, etc. ===

    @abstractmethod
    def put(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a checkpoint."""
        raise NotImplementedError("put method must be implemented")

    @abstractmethod
    def get(self, config: dict[str, Any]) -> list[Message]:
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

    def delete(self, config: dict[str, Any]) -> None:
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
        """List all threads."""
        raise NotImplementedError("list_threads method must be implemented")

    def cleanup(
        self,
        config: dict[str, Any],
    ) -> None:
        """Cleanup resources if needed, This will delete all checkpoints for a thread."""
        raise NotImplementedError("cleanup method must be implemented")
