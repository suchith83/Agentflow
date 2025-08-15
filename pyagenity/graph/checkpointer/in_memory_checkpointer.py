from typing import Any

from pyagenity.graph.state import AgentState
from pyagenity.graph.state.execution_state import ExecutionState
from pyagenity.graph.utils import Message

from .base_checkpointer import BaseCheckpointer


class InMemoryCheckpointer(BaseCheckpointer):
    """In-memory checkpointer that persists combined AgentState (including execution metadata)."""

    def __init__(self):
        # Simulate tables with dicts/lists
        self._threads: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[Message]] = {}
        self._states: dict[str, AgentState] = {}
        # Note: _execution_states is kept for compatibility but execution state
        # is now embedded in AgentState.execution_meta

    def put(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._messages[thread_id] = messages.copy()
        if metadata:
            self._threads.setdefault(thread_id, {})["metadata"] = metadata

    def get(self, config: dict[str, Any]) -> list[Message]:
        thread_id = config.get("thread_id", "default")
        return self._messages.get(thread_id, []).copy()

    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        thread_id = config.get("thread_id", "default")
        messages = self._messages.get(thread_id, [])
        if search:
            messages = [m for m in messages if search in m.content]
        if offset is not None:
            messages = messages[offset:]
        if limit is not None:
            messages = messages[:limit]
        return messages.copy()

    def delete(self, config: dict[str, Any]) -> None:
        thread_id = config.get("thread_id", "default")
        self._messages.pop(thread_id, None)

    # === PRIMARY API: Combined State Management ===

    def put_state(
        self,
        config: dict[str, Any],
        state: AgentState,
    ) -> None:
        """Store complete AgentState (including execution metadata) atomically."""
        thread_id = config.get("thread_id", "default")
        self._states[thread_id] = state

    def get_state(self, config: dict[str, Any]) -> AgentState | None:
        """Get the complete AgentState (including execution metadata)."""
        thread_id = config.get("thread_id", "default")
        return self._states.get(thread_id)

    def clear_state(
        self,
        config: dict[str, Any],
    ) -> None:
        """Clear the complete AgentState for the given config."""
        thread_id = config.get("thread_id", "default")
        self._states.pop(thread_id, None)

    # === LEGACY API: For Backward Compatibility ===

    def update_state(
        self,
        config: dict[str, Any],
        state: AgentState,
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._states[thread_id] = state

    def put_thread(
        self,
        config: dict[str, Any],
        thread_info: dict[str, Any],
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._threads[thread_id] = thread_info.copy()

    def get_thread(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        thread_id = config.get("thread_id", "default")
        return self._threads.get(thread_id)

    def list_threads(
        self,
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        threads = list(self._threads.values())
        if search:
            threads = [t for t in threads if search in str(t)]
        if offset is not None:
            threads = threads[offset:]
        if limit is not None:
            threads = threads[:limit]
        return [t.copy() for t in threads]

    def cleanup(
        self,
        config: dict[str, Any],
    ) -> None:
        thread_id = config.get("thread_id", "default")
        self._messages.pop(thread_id, None)
        self._states.pop(thread_id, None)
        self._threads.pop(thread_id, None)

    def put_execution_state(
        self,
        config: dict[str, Any],
        execution_state: ExecutionState,
    ) -> None:
        """Store execution state for pause/resume functionality.

        This method is kept for compatibility but execution state is now embedded
        in AgentState.execution_meta. This method updates the existing AgentState
        or creates a minimal one if it doesn't exist.
        """
        thread_id = config.get("thread_id", "default")
        execution_state.thread_id = thread_id

        # Get or create AgentState to store the execution metadata
        state = self._states.get(thread_id)
        if state:
            # Update execution metadata in existing state
            state.execution_meta = execution_state
        else:
            # Create minimal state with execution metadata
            state = AgentState()
            state.execution_meta = execution_state

        self._states[thread_id] = state

    def get_execution_state(
        self,
        config: dict[str, Any],
    ) -> ExecutionState | None:
        """Retrieve execution state for pause/resume functionality.

        This method is kept for compatibility but execution state is now embedded
        in AgentState.execution_meta.
        """
        thread_id = config.get("thread_id", "default")
        state = self._states.get(thread_id)
        if state:
            return state.execution_meta
        return None

    def clear_execution_state(
        self,
        config: dict[str, Any],
    ) -> None:
        """Clear execution state when execution completes or errors.

        This method is kept for compatibility. It resets the execution metadata
        in the AgentState but keeps the rest of the state intact.
        """
        thread_id = config.get("thread_id", "default")
        state = self._states.get(thread_id)
        if state:
            # Reset execution metadata to initial state
            state.execution_meta = ExecutionState(current_node="__start__")
