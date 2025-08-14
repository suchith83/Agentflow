from dataclasses import dataclass
from typing_extensions import Annotated
from pyagenity.graph.utils import Message
from pyagenity.graph.utils import add_messages


@dataclass
class AgentState:
    """Common state schema that includes messages and context."""

    context: Annotated[list[Message], add_messages]
    active_node: str
    step: int
