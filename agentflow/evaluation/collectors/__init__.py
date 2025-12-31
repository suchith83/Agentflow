"""
Event collectors for capturing execution data during agent runs.

This module provides collectors that capture various aspects of agent execution:
    - TrajectoryCollector: Captures tool calls, node visits, and execution path
    - EventCollector: General-purpose event collection
"""

from agentflow.evaluation.collectors.trajectory_collector import (
    EventCollector,
    TrajectoryCollector,
)

__all__ = [
    "EventCollector",
    "TrajectoryCollector",
]
