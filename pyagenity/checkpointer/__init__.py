"""
Checkpointer adapters for agent state persistence in PyAgenity.

This module exposes unified checkpointing interfaces for agent graphs, supporting
in-memory and Postgres-backed persistence. PgCheckpointer is only exported if its
dependencies (asyncpg, redis) are available.

Exports:
    BaseCheckpointer: Abstract base class for checkpointing implementations.
    InMemoryCheckpointer: In-memory checkpointing for development/testing.
    PgCheckpointer: Postgres+Redis checkpointing (optional, requires extras).

Usage:
    PgCheckpointer requires: pip install pyagenity[pg_checkpoint]
"""

from .base_checkpointer import BaseCheckpointer
from .in_memory_checkpointer import InMemoryCheckpointer


# Conditionally import PgCheckpointer only if dependencies are available
try:
    from .pg_checkpointer import PgCheckpointer

    __all__ = [
        "BaseCheckpointer",
        "InMemoryCheckpointer",
        "PgCheckpointer",
    ]
except ImportError:
    # PgCheckpointer requires asyncpg and redis
    # Install with: pip install pyagenity[pg_checkpoint]
    __all__ = [
        "BaseCheckpointer",
        "InMemoryCheckpointer",
    ]
