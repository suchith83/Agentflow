"""Shared mixins for graph/node handler classes.

These mixins provide light structure for logging, configuration, and
event publishing without changing runtime behavior. Handlers can
inherit one or more to keep responsibilities explicit.
"""

from __future__ import annotations

import logging
from typing import Any


class BaseLoggingMixin:
    """Adds small logging helpers to handler classes."""

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(getattr(self, "__module__", __name__))

    def _log_start(self, msg: str, *args: Any) -> None:
        self._logger.info(msg, *args)

    def _log_debug(self, msg: str, *args: Any) -> None:
        self._logger.debug(msg, *args)

    def _log_error(self, msg: str, *args: Any) -> None:
        self._logger.error(msg, *args)


class InterruptConfigMixin:
    """Holds interrupt configuration for graph-level handlers."""

    interrupt_before: list[str] | None
    interrupt_after: list[str] | None

    def _set_interrupts(
        self,
        interrupt_before: list[str] | None,
        interrupt_after: list[str] | None,
    ) -> None:
        self.interrupt_before = interrupt_before or []
        self.interrupt_after = interrupt_after or []
