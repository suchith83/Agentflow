from dataclasses import dataclass, field
from typing import Annotated

from pyagenity.graph.utils import ExecutionState, Message, add_messages


@dataclass
class AgentState:
    """Common state schema that includes messages and context."""

    context: Annotated[list[Message], add_messages] = field(default_factory=list)
    context_summary: str | None = None
    active_node: str = ""
    execution_state: ExecutionState = ExecutionState.IDLE
    step: int = 0
