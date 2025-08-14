"""
State management for AgentGraph.

Provides state schemas, reducers, and context management similar to LangGraph.
"""

from typing import Any, Dict, List, Optional, TypeVar, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime

T = TypeVar("T")


# Base state interface
class State(TypedDict):
    """Base state schema for AgentGraph."""

    pass
