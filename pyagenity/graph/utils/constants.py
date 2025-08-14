from enum import StrEnum
from typing import Literal


# Special node names for graph execution flow
START: Literal["__start__"] = "__start__"
END: Literal["__end__"] = "__end__"


# Message storage levels
class StorageLevel:
    ALL = "all"  # Save everything including tool calls
    MEDIUM = "medium"  # Only AI and human messages
    LOW = "low"  # Only first human and last AI message


# Graph execution states
class ExecutionState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    ABORTED = "aborted"
    IDLE = "idle"


class ResponseGranularity(StrEnum):
    FULL = "full"  # State, Latest Messages
    PARTIAL = "partial"  # Context, Summary, Latest Messages
    LOW = "low"  # Only Latest Messages
