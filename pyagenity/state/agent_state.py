from dataclasses import dataclass, field
from typing import Annotated, Any, TypeVar

from pyagenity.state.execution_state import ExecutionState as ExecMeta
from pyagenity.utils import START, Message, add_messages


# Generic type variable for state subclassing
StateT = TypeVar("StateT", bound="AgentState")


@dataclass
class AgentState:
    """Common state schema that includes messages, context and internal execution metadata.

    This class can be subclassed to add application-specific fields while maintaining
    compatibility with the PyAgenity framework. All internal execution metadata
    is preserved through subclassing.

    Notes:
    - `execution_meta` contains internal-only execution progress and interrupt info.
    - Users may subclass `AgentState` to add application fields; internal exec meta remains
      available to the runtime and will be persisted with the state.
    - When subclassing, add your fields but keep the core fields intact.

    Example:
        @dataclass
        class MyCustomState(AgentState):
            user_data: dict = field(default_factory=dict)
            custom_field: str = "default"
    """

    context: Annotated[list[Message], add_messages] = field(default_factory=list)
    context_summary: str | None = None
    # Internal execution metadata (kept private-ish but accessible to runtime)
    execution_meta: ExecMeta = field(default_factory=lambda: ExecMeta(current_node=START))

    def to_dict(self, include_internal: bool = False) -> dict[str, Any]:
        """Serialize state to dict.

        By default internal execution metadata is excluded unless include_internal=True.
        """
        d = {
            "context": [m.to_dict() for m in self.context],
            "context_summary": self.context_summary,
            "active_node": self.execution_meta.current_node,
            "step": self.execution_meta.step,
        }
        if include_internal:
            d["execution_meta"] = self.execution_meta.to_dict()
        return d

    # Convenience delegation methods for execution meta so callers can use the same API
    def set_interrupt(self, node: str, reason: str, status, data: dict | None = None) -> None:
        self.execution_meta.set_interrupt(node, reason, status, data)

    def clear_interrupt(self) -> None:
        self.execution_meta.clear_interrupt()

    def is_running(self) -> bool:
        return self.execution_meta.is_running()

    def is_interrupted(self) -> bool:
        return self.execution_meta.is_interrupted()

    def advance_step(self) -> None:
        self.execution_meta.advance_step()

    def set_current_node(self, node: str) -> None:
        self.execution_meta.set_current_node(node)

    def complete(self) -> None:
        self.execution_meta.complete()

    def error(self, error_msg: str) -> None:
        self.execution_meta.error(error_msg)
