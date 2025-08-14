from dataclasses import dataclass
from typing import Optional
from typing_extensions import Annotated
from pyagenity.graph.utils import Message
from pyagenity.graph.utils import add_messages
from pyagenity.graph.utils.constants import ExecutionState


@dataclass
class AgentState:
    """Common state schema that includes messages and context."""

    context: Annotated[list[Message], add_messages] = []
    context_summary: Optional[str] = None
    active_node: str = ""
    execution_state: ExecutionState = ExecutionState.IDLE
    step: int = 0
