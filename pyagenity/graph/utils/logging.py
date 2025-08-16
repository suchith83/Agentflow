"""
Comprehensive logging framework for PyAgenity with security, performance, and debugging support.

This module provides structured logging with:
- Security event logging with sanitization
- Performance metrics tracking
- Request tracing with correlation IDs
- Debug logging with proper levels
- Sensitive data masking
"""

import asyncio
import functools
import hashlib
import logging
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Union

# Thread-local storage for correlation IDs
_local = threading.local()

# Sensitive data patterns for masking
SENSITIVE_PATTERNS = [
    (
        re.compile(
            r"(api[_-]?key|token|password|secret|auth)[\"']?\s*[:=]\s*[\"']?([^\"'\s,}]+)",
            re.IGNORECASE,
        ),
        r"\1=***MASKED***",
    ),
    (re.compile(r"(bearer\s+)([a-zA-Z0-9._-]+)", re.IGNORECASE), r"\1***MASKED***"),
    (
        re.compile(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", re.IGNORECASE),
        r"***EMAIL_MASKED***",
    ),
    (re.compile(r"(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})", re.IGNORECASE), r"***CARD_MASKED***"),
]


class SecurityLogger:
    """Security-focused logger for sensitive operations."""

    def __init__(self, logger_name: str = "pyagenity.security"):
        self.logger = logging.getLogger(logger_name)

    def log_security_event(self, event_type: str, details: dict[str, Any], severity: str = "INFO"):
        """Log security events with proper sanitization."""
        sanitized_details = self._sanitize_data(details)
        correlation_id = get_correlation_id()

        self.logger.log(
            getattr(logging, severity.upper()),
            f"SECURITY_EVENT: {event_type}",
            extra={
                "correlation_id": correlation_id,
                "event_type": event_type,
                "details": sanitized_details,
                "security_event": True,
            },
        )

    def log_input_validation_failure(self, input_type: str, validation_error: str, input_data: Any):
        """Log input validation failures."""
        self.log_security_event(
            "INPUT_VALIDATION_FAILURE",
            {
                "input_type": input_type,
                "validation_error": validation_error,
                "input_hash": self._hash_data(str(input_data)),
            },
            "WARNING",
        )

    def log_authentication_event(self, event_type: str, user_id: str | None, success: bool):
        """Log authentication events."""
        self.log_security_event(
            f"AUTH_{event_type}",
            {"user_id": user_id, "success": success, "timestamp": time.time()},
            "INFO" if success else "WARNING",
        )

    @staticmethod
    def _sanitize_data(data: Any) -> Any:
        """Sanitize sensitive data for logging."""
        if isinstance(data, str):
            sanitized = data
            for pattern, replacement in SENSITIVE_PATTERNS:
                sanitized = pattern.sub(replacement, sanitized)
            return sanitized
        if isinstance(data, dict):
            return {k: SecurityLogger._sanitize_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [SecurityLogger._sanitize_data(item) for item in data]
        return data

    @staticmethod
    def _hash_data(data: str) -> str:
        """Create a hash of data for logging purposes."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class PerformanceLogger:
    """Performance-focused logger for metrics and bottleneck analysis."""

    def __init__(self, logger_name: str = "pyagenity.performance"):
        self.logger = logging.getLogger(logger_name)

    def log_execution_time(
        self, operation: str, duration: float, details: dict[str, Any] | None = None
    ):
        """Log execution time for operations."""
        correlation_id = get_correlation_id()

        self.logger.info(
            "PERFORMANCE: %s completed in %.4fs",
            operation,
            duration,
            extra={
                "correlation_id": correlation_id,
                "operation": operation,
                "duration": duration,
                "details": details or {},
                "performance_metric": True,
            },
        )

    def log_memory_usage(
        self, operation: str, memory_mb: float, details: dict[str, Any] | None = None
    ):
        """Log memory usage for operations."""
        correlation_id = get_correlation_id()

        self.logger.info(
            "MEMORY: %s used %.2fMB",
            operation,
            memory_mb,
            extra={
                "correlation_id": correlation_id,
                "operation": operation,
                "memory_mb": memory_mb,
                "details": details or {},
                "memory_metric": True,
            },
        )

    def log_resource_usage(self, resource_type: str, usage_count: int, limit: int | None = None):
        """Log resource usage (connections, threads, etc.)."""
        correlation_id = get_correlation_id()

        warning_level = False
        if limit and usage_count > limit * 0.8:  # Warn at 80% of limit
            warning_level = True

        limit_str = f"/{limit}" if limit else ""
        self.logger.log(
            logging.WARNING if warning_level else logging.INFO,
            "RESOURCE: %s usage: %d%s",
            resource_type,
            usage_count,
            limit_str,
            extra={
                "correlation_id": correlation_id,
                "resource_type": resource_type,
                "usage_count": usage_count,
                "limit": limit,
                "resource_metric": True,
            },
        )


class DebugLogger:
    """Debug-focused logger for development and troubleshooting."""

    def __init__(self, logger_name: str = "pyagenity.debug"):
        self.logger = logging.getLogger(logger_name)

    def log_graph_execution(self, current_node: str, step: int, state_info: dict[str, Any]):
        """Log detailed graph execution information."""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        correlation_id = get_correlation_id()

        self.logger.debug(
            "GRAPH_EXECUTION: node=%s, step=%d",
            current_node,
            step,
            extra={
                "correlation_id": correlation_id,
                "current_node": current_node,
                "step": step,
                "state_info": SecurityLogger._sanitize_data(state_info),
                "graph_execution": True,
            },
        )

    def log_node_execution(
        self, node_name: str, input_data: Any, output_data: Any, duration: float
    ):
        """Log detailed node execution information."""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        correlation_id = get_correlation_id()

        self.logger.debug(
            "NODE_EXECUTION: %s completed in %.4fs",
            node_name,
            duration,
            extra={
                "correlation_id": correlation_id,
                "node_name": node_name,
                "input_hash": SecurityLogger._hash_data(str(input_data)),
                "output_hash": SecurityLogger._hash_data(str(output_data)),
                "duration": duration,
                "node_execution": True,
            },
        )

    def log_state_change(self, change_type: str, old_state: Any, new_state: Any):
        """Log state changes for debugging."""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        correlation_id = get_correlation_id()

        self.logger.debug(
            "STATE_CHANGE: %s",
            change_type,
            extra={
                "correlation_id": correlation_id,
                "change_type": change_type,
                "old_state_hash": SecurityLogger._hash_data(str(old_state)),
                "new_state_hash": SecurityLogger._hash_data(str(new_state)),
                "state_change": True,
            },
        )


# Correlation ID management
def get_correlation_id() -> str:
    """Get the current correlation ID, creating one if none exists."""
    if not hasattr(_local, "correlation_id"):
        _local.correlation_id = str(uuid.uuid4())
    return _local.correlation_id


def set_correlation_id(correlation_id: str):
    """Set the correlation ID for the current thread."""
    _local.correlation_id = correlation_id


@contextmanager
def correlation_context(correlation_id: str | None = None):
    """Context manager for correlation ID scope."""
    old_id = getattr(_local, "correlation_id", None)
    new_id = correlation_id or str(uuid.uuid4())

    try:
        set_correlation_id(new_id)
        yield new_id
    finally:
        if old_id:
            set_correlation_id(old_id)
        else:
            delattr(_local, "correlation_id")


# Performance monitoring decorators
def monitor_performance(operation_name: str | None = None):
    """Decorator to monitor function performance."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            perf_logger = PerformanceLogger()

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                perf_logger.log_execution_time(op_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                perf_logger.log_execution_time(f"{op_name}_ERROR", duration, {"error": str(e)})
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            perf_logger = PerformanceLogger()

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                perf_logger.log_execution_time(op_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                perf_logger.log_execution_time(f"{op_name}_ERROR", duration, {"error": str(e)})
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Logger configuration
def configure_logging(debug: bool = False, log_file: str | None = None):
    """Configure logging for the entire PyAgenity framework."""
    # Set log levels
    root_logger = logging.getLogger("pyagenity")
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(correlation_id)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Ensure correlation_id is in log records
    class CorrelationFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, "correlation_id"):
                record.correlation_id = get_correlation_id()
            return True

    correlation_filter = CorrelationFilter()
    for handler in root_logger.handlers:
        handler.addFilter(correlation_filter)


# Singleton instances for easy access
security_logger = SecurityLogger()
performance_logger = PerformanceLogger()
debug_logger = DebugLogger()
