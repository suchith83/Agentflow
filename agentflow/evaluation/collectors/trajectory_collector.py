"""
Trajectory collector for capturing execution paths during agent runs.

This module provides the TrajectoryCollector class which captures the execution
trajectory from EventModel events during graph execution.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from agentflow.evaluation.eval_set import StepType, ToolCall, TrajectoryStep
from agentflow.publisher.events import ContentType, Event, EventType

if TYPE_CHECKING:
    from agentflow.publisher.events import EventModel
    from agentflow.state import Message

logger = logging.getLogger("agentflow.evaluation")


class TrajectoryCollector:
    """Collects execution trajectory from graph events.

    This class captures the sequence of nodes visited, tools called, and
    messages exchanged during agent graph execution. It can be used as an
    event callback to track the actual execution path for comparison with
    expected trajectories.

    Attributes:
        trajectory: Complete execution trajectory as TrajectoryStep objects.
        tool_calls: List of tool calls made during execution.
        node_visits: List of node names visited in order.
        messages: List of messages captured during execution.
        events: All events captured (for debugging).
        start_time: When collection started.
        end_time: When collection ended.

    Example:
        ```python
        collector = TrajectoryCollector()

        # Use as callback during graph execution
        result = await graph.invoke(state, config={"callbacks": [collector.on_event]})

        # Analyze collected trajectory
        print(f"Visited nodes: {collector.node_visits}")
        print(f"Tool calls: {collector.tool_calls}")
        ```
    """

    def __init__(self, capture_all_events: bool = False):
        """Initialize the trajectory collector.

        Args:
            capture_all_events: If True, store all events for debugging.
        """
        self.trajectory: list[TrajectoryStep] = []
        self.tool_calls: list[ToolCall] = []
        self.node_visits: list[str] = []
        self.messages: list[dict[str, Any]] = []
        self.events: list[EventModel] = []
        self.capture_all_events = capture_all_events
        self.start_time: float | None = None
        self.end_time: float | None = None
        self._seen_tool_starts: set[str] = set()  # Track tool call IDs

    def reset(self) -> None:
        """Reset the collector to initial state."""
        self.trajectory.clear()
        self.tool_calls.clear()
        self.node_visits.clear()
        self.messages.clear()
        self.events.clear()
        self._seen_tool_starts.clear()
        self.start_time = None
        self.end_time = None

    async def on_event(self, event: EventModel) -> None:
        """Process an incoming event and update trajectory.

        This method should be passed as a callback to graph execution.

        Args:
            event: The event to process.
        """
        if self.start_time is None:
            self.start_time = time.time()

        if self.capture_all_events:
            self.events.append(event)

        # Process based on event type
        if event.event == Event.NODE_EXECUTION:
            self._process_node_event(event)
        elif event.event == Event.TOOL_EXECUTION:
            self._process_tool_event(event)
        elif event.event == Event.GRAPH_EXECUTION:
            self._process_graph_event(event)

    def on_event_sync(self, event: EventModel) -> None:
        """Synchronous version of on_event for non-async contexts.

        Args:
            event: The event to process.
        """
        if self.start_time is None:
            self.start_time = time.time()

        if self.capture_all_events:
            self.events.append(event)

        if event.event == Event.NODE_EXECUTION:
            self._process_node_event(event)
        elif event.event == Event.TOOL_EXECUTION:
            self._process_tool_event(event)
        elif event.event == Event.GRAPH_EXECUTION:
            self._process_graph_event(event)

    def _process_node_event(self, event: EventModel) -> None:
        """Process a node execution event."""
        if event.event_type == EventType.START:
            node_name = event.node_name or event.data.get("node_name", "")
            if node_name:
                self.node_visits.append(node_name)
                event_type_str = (
                    event.event_type.value
                    if hasattr(event.event_type, "value")
                    else str(event.event_type)
                )
                self.trajectory.append(
                    TrajectoryStep.node(
                        name=node_name,
                        timestamp=event.timestamp,
                        event_type=event_type_str,
                    )
                )
                logger.debug("Node visit recorded: %s", node_name)

    def _process_tool_event(self, event: EventModel) -> None:
        """Process a tool execution event."""
        # Only capture tool calls on START to avoid duplicates
        if event.event_type == EventType.START:
            tool_call_id = event.data.get("tool_call_id", "")
            function_name = event.data.get("function_name", "")
            args = event.data.get("args", {})

            # Avoid duplicate tool calls
            if tool_call_id and tool_call_id in self._seen_tool_starts:
                return
            if tool_call_id:
                self._seen_tool_starts.add(tool_call_id)

            if function_name:
                tool_call = ToolCall(
                    name=function_name,
                    args=args,
                    call_id=tool_call_id,
                )
                self.tool_calls.append(tool_call)
                self.trajectory.append(
                    TrajectoryStep.tool(
                        name=function_name,
                        args=args,
                        timestamp=event.timestamp,
                        tool_call_id=tool_call_id,
                    )
                )
                logger.debug("Tool call recorded: %s(%s)", function_name, args)

        # Capture tool results on END
        elif event.event_type == EventType.END:
            tool_call_id = event.data.get("tool_call_id", "")
            # Update the tool call with result if we have one
            if tool_call_id:
                for tc in reversed(self.tool_calls):
                    if tc.call_id == tool_call_id:
                        tc.result = event.data.get("result") or event.data.get("message")
                        break

    def _process_graph_event(self, event: EventModel) -> None:
        """Process a graph execution event."""
        if event.event_type == EventType.END:
            self.end_time = time.time()

    def get_tool_names(self) -> list[str]:
        """Get list of tool names called in order."""
        return [tc.name for tc in self.tool_calls]

    def get_trajectory_steps(self, step_type: StepType | None = None) -> list[TrajectoryStep]:
        """Get trajectory steps, optionally filtered by type.

        Args:
            step_type: If provided, only return steps of this type.

        Returns:
            List of trajectory steps.
        """
        if step_type is None:
            return self.trajectory
        return [s for s in self.trajectory if s.step_type == step_type]

    @property
    def duration(self) -> float:
        """Get duration of the collected execution in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert collector data to dictionary for serialization."""
        return {
            "trajectory": [s.model_dump() for s in self.trajectory],
            "tool_calls": [tc.model_dump() for tc in self.tool_calls],
            "node_visits": self.node_visits,
            "duration_seconds": self.duration,
        }

    def __repr__(self) -> str:
        return (
            f"TrajectoryCollector("
            f"nodes={len(self.node_visits)}, "
            f"tools={len(self.tool_calls)}, "
            f"steps={len(self.trajectory)})"
        )


class EventCollector:
    """Simple collector that stores all events.

    This is useful for debugging and detailed analysis of execution flow.

    Attributes:
        events: List of all captured events.
    """

    def __init__(self):
        """Initialize the event collector."""
        self.events: list[EventModel] = []

    def reset(self) -> None:
        """Reset the collector."""
        self.events.clear()

    async def on_event(self, event: EventModel) -> None:
        """Capture an event."""
        self.events.append(event)

    def on_event_sync(self, event: EventModel) -> None:
        """Synchronous version of on_event."""
        self.events.append(event)

    def filter_by_event(self, event_type: Event) -> list[EventModel]:
        """Filter events by event source type."""
        return [e for e in self.events if e.event == event_type]

    def filter_by_event_type(self, event_type: EventType) -> list[EventModel]:
        """Filter events by event phase type."""
        return [e for e in self.events if e.event_type == event_type]

    def filter_by_node(self, node_name: str) -> list[EventModel]:
        """Filter events by node name."""
        return [e for e in self.events if e.node_name == node_name]

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"EventCollector(events={len(self.events)})"
