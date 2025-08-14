"""
Command API for AgentGraph.

Allows combining state updates with control flow similar to LangGraph's Command.
"""

from typing import Any, Dict, Generic, TypeVar, Optional, Literal


T = TypeVar("T")


class Command(Generic[T]):
    """
    Command object that combines state updates with control flow.

    Similar to LangGraph's Command API, allows nodes to both update state
    and direct graph execution to specific nodes.
    """

    PARENT = "PARENT"

    def __init__(
        self,
        update: Optional[Dict[str, Any]] | Optional[MessagesState] = None,
        goto: Optional[T] = None,
        graph: Optional[str] = None,
    ):
        """
        Initialize a Command.

        Args:
            update: Dictionary of state updates to apply
            goto: Next node to execute (can be node name or END)
            graph: Which graph to navigate to (None for current, PARENT for parent)
        """
        self.update = update or {}
        self.goto = goto
        self.graph = graph

    def __repr__(self) -> str:
        return f"Command(update={self.update}, goto={self.goto}, graph={self.graph})"


# Type aliases for common command return types
CommandLiteral = Command[Literal]
CommandStr = Command[str]


def create_command(
    update: Optional[Dict[str, Any]] = None,
    goto: Optional[str] = None,
    graph: Optional[str] = None,
) -> Command[str]:
    """Utility function to create a Command."""
    return Command(update=update, goto=goto, graph=graph)
