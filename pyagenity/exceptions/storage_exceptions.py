"""Structured exception taxonomy for persistence & runtime layers.

These exceptions let higher-level orchestration decide retry / fail-fast logic
instead of relying on broad ``except Exception`` blocks.
"""

from __future__ import annotations


class StorageError(Exception):
    """Base class for non-retryable storage layer errors."""


class TransientStorageError(StorageError):
    """Retryable storage error (connection drops, timeouts)."""


class SerializationError(StorageError):
    """Raised when (de)serialization of state/messages fails deterministically."""


class SchemaVersionError(StorageError):
    """Raised when schema version detection or migration application fails."""


class MetricsError(Exception):
    """Raised when metrics emission fails (should normally be swallowed/logged)."""
