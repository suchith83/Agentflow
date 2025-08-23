import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, TypeVar

from pyagenity.state.execution_state import ExecutionState as ExecMeta
from pyagenity.utils import START, Message, add_messages


# Generic type variable for state subclassing
StateT = TypeVar("StateT", bound="AgentState")

logger = logging.getLogger(__name__)


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
        """Serialize state to dict, including all dataclass fields (custom fields too)."""
        import dataclasses  # noqa: PLC0415

        d = dataclasses.asdict(self)
        # Overwrite context with serialized messages
        d["context"] = [m.to_dict() for m in self.context]
        # Optionally add execution_meta as dict
        if include_internal:
            d["execution_meta"] = self.execution_meta.to_dict()
        else:
            # Remove execution_meta if not requested
            d.pop("execution_meta", None)
        # Add convenience fields for compatibility
        d["active_node"] = self.execution_meta.current_node
        d["step"] = self.execution_meta.step
        return d

    # Convenience delegation methods for execution meta so callers can use the same API
    def set_interrupt(self, node: str, reason: str, status, data: dict | None = None) -> None:
        logger.debug("Setting interrupt at node '%s' with reason: %s", node, reason)
        self.execution_meta.set_interrupt(node, reason, status, data)

    def clear_interrupt(self) -> None:
        logger.debug("Clearing interrupt")
        self.execution_meta.clear_interrupt()

    def is_running(self) -> bool:
        running = self.execution_meta.is_running()
        logger.debug("State is_running: %s", running)
        return running

    def is_interrupted(self) -> bool:
        interrupted = self.execution_meta.is_interrupted()
        logger.debug("State is_interrupted: %s", interrupted)
        return interrupted

    def advance_step(self) -> None:
        old_step = self.execution_meta.step
        self.execution_meta.advance_step()
        logger.debug("Advanced step from %d to %d", old_step, self.execution_meta.step)

    def set_current_node(self, node: str) -> None:
        old_node = self.execution_meta.current_node
        self.execution_meta.set_current_node(node)
        logger.debug("Changed current node from '%s' to '%s'", old_node, node)

    def complete(self) -> None:
        logger.info("Marking state as completed")
        self.execution_meta.complete()

    def error(self, error_msg: str) -> None:
        logger.error("Setting state error: %s", error_msg)
        self.execution_meta.error(error_msg)
