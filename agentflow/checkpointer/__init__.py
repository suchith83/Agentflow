"""
Checkpointer adapters for agent state persistence in agentflow.

This module exposes unified checkpointing interfaces for agent graphs, supporting
in-memory and Postgres-backed persistence. PgCheckpointer is only exported if its
dependencies (asyncpg, redis) are available.

Exports:
    BaseCheckpointer: Abstract base class for checkpointing implementations.
    InMemoryCheckpointer: In-memory checkpointing for development/testing.
    PgCheckpointer: Postgres+Redis checkpointing (optional, requires extras).

Usage:
    PgCheckpointer requires: pip install 10xscale-agentflow[pg_checkpoint]
"""

from .base_checkpointer import BaseCheckpointer
from .in_memory_checkpointer import InMemoryCheckpointer
from .pg_checkpointer import PgCheckpointer


__all__ = [
    "BaseCheckpointer",
    "InMemoryCheckpointer",
    "PgCheckpointer",
]
