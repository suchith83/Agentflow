"""
Command API for AgentGraph.

Allows combining state updates with control flow similar to LangGraph's Command.
"""

from typing import TYPE_CHECKING, TypeVar, Union

from litellm.types.utils import ModelResponse


if TYPE_CHECKING:
    # Import only for type checking to avoid circular imports at runtime
    from pyagenity.state import AgentState


StateT = TypeVar("StateT", bound="AgentState")


class Command[StateT: AgentState]:
    """
    Command object that combines state updates with control flow.

    Similar to LangGraph's Command API, allows nodes to both update state
    and direct graph execution to specific nodes.
    """

    PARENT = "PARENT"

    def __init__(
        self,
        update: Union["StateT", None, ModelResponse] = None,
        goto: str | None = None,
        graph: str | None = None,
    ):
        """
        Initialize a Command.

        Args:
            update: Dictionary of state updates to apply
            goto: Next node to execute (can be node name or END)
            graph: Which graph to navigate to (None for current, PARENT for parent)
        """
        self.update = update
        self.goto = goto
        self.graph = graph

    def __repr__(self) -> str:
        return f"Command(update={self.update}, goto={self.goto}, graph={self.graph})"
