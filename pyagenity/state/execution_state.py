"""
Execution state management for graph execution with pause/resume functionality.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionStatus(Enum):
    """Status of graph execution."""

    RUNNING = "running"
    INTERRUPTED_BEFORE = "interrupted_before"
    INTERRUPTED_AFTER = "interrupted_after"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ExecutionState:
    """
    Tracks the internal execution state of a graph.

    This class manages the execution progress, interrupt status, and internal
    data that should not be exposed to users.
    """

    # Core execution tracking
    current_node: str
    step: int = 0
    status: ExecutionStatus = ExecutionStatus.RUNNING

    # Interrupt management
    interrupted_node: str | None = None
    interrupt_reason: str | None = None
    interrupt_data: dict[str, Any] | None = None

    # Thread/session identification
    thread_id: str | None = None

    # Internal execution data (hidden from user)
    _internal_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_node": self.current_node,
            "step": self.step,
            "status": self.status.value,
            "interrupted_node": self.interrupted_node,
            "interrupt_reason": self.interrupt_reason,
            "interrupt_data": self.interrupt_data,
            "thread_id": self.thread_id,
            "_internal_data": self._internal_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionState":
        """Create from dictionary for deserialization."""
        return cls(
            current_node=data["current_node"],
            step=data.get("step", 0),
            status=ExecutionStatus(data.get("status", "running")),
            interrupted_node=data.get("interrupted_node"),
            interrupt_reason=data.get("interrupt_reason"),
            interrupt_data=data.get("interrupt_data"),
            thread_id=data.get("thread_id"),
            _internal_data=data.get("_internal_data", {}),
        )

    def set_interrupt(
        self, node: str, reason: str, status: ExecutionStatus, data: dict[str, Any] | None = None
    ) -> None:
        """Set interrupt state."""
        self.interrupted_node = node
        self.interrupt_reason = reason
        self.status = status
        self.interrupt_data = data

    def clear_interrupt(self) -> None:
        """Clear interrupt state and resume execution."""
        self.interrupted_node = None
        self.interrupt_reason = None
        self.interrupt_data = None
        self.status = ExecutionStatus.RUNNING

    def is_interrupted(self) -> bool:
        """Check if execution is currently interrupted."""
        return self.status in [
            ExecutionStatus.INTERRUPTED_BEFORE,
            ExecutionStatus.INTERRUPTED_AFTER,
        ]

    def advance_step(self) -> None:
        """Advance to next step."""
        self.step += 1

    def set_current_node(self, node: str) -> None:
        """Update current node."""
        self.current_node = node

    def complete(self) -> None:
        """Mark execution as completed."""
        self.status = ExecutionStatus.COMPLETED

    def error(self, error_msg: str) -> None:
        """Mark execution as errored."""
        self.status = ExecutionStatus.ERROR
        self._internal_data["error"] = error_msg

    def is_running(self) -> bool:
        """Check if execution is currently running."""
        return self.status == ExecutionStatus.RUNNING
