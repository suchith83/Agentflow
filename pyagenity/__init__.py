"""Top-level pyagenity package exports.

Expose common classes/functions so consumers can import directly from
`pyagenity` instead of from deep package paths.
"""

# Re-export commonly used graph symbols (avoid heavy optional imports at package import time)
from .graph.state_graph import StateGraph
from .graph.edge import Edge
from .graph.node import Node
from .graph.tool_node import ToolNode

from .graph.checkpointer import BaseCheckpointer, BaseStore, InMemoryCheckpointer
from .graph.state import AgentState, ExecutionStatus
from .graph.utils import DependencyContainer, Message, StreamChunk, ResponseGranularity

# Provide alias matching README
Graph = StateGraph

__all__ = [
    "Graph",
    "StateGraph",
    "Edge",
    "Node",
    "ToolNode",
    "BaseCheckpointer",
    "BaseStore",
    "InMemoryCheckpointer",
    "AgentState",
    "ExecutionStatus",
    "DependencyContainer",
    "Message",
    "StreamChunk",
    "ResponseGranularity",
]
