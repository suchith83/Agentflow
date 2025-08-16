"""
Security validation and hardening module for PyAgenity.

This module provides comprehensive security measures including:
- Input validation and sanitization
- LLM prompt injection detection
- Resource usage limiting
- Authentication and authorization framework
- Safe data serialization
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pyagenity.graph.utils.logging import security_logger


# Security configuration constants
MAX_INPUT_LENGTH = 50000  # Maximum input string length
MAX_CONTEXT_MESSAGES = 1000  # Maximum messages in context
MAX_EXECUTION_STEPS = 100  # Maximum graph execution steps
MAX_THREAD_ID_LENGTH = 255  # Maximum thread ID length
MAX_NODE_NAME_LENGTH = 100  # Maximum node name length

# Dangerous patterns that could indicate injection attacks
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"forget\s+everything", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"assistant\s*:\s*i\s+(will|can|should)", re.IGNORECASE),
    re.compile(r"\\n\\n.*?system\s*:", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+are", re.IGNORECASE),
    re.compile(r"pretend\s+to\s+be", re.IGNORECASE),
    re.compile(r"role\s*:\s*system", re.IGNORECASE),
    re.compile(r"</?\s*(script|iframe|object|embed)", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
]

# Allowed characters for different input types
THREAD_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
NODE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
CONFIG_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


@dataclass
class SecurityConfig:
    """Security configuration for the framework."""

    # Input validation settings
    max_input_length: int = MAX_INPUT_LENGTH
    max_context_messages: int = MAX_CONTEXT_MESSAGES
    max_execution_steps: int = MAX_EXECUTION_STEPS
    max_thread_id_length: int = MAX_THREAD_ID_LENGTH
    max_node_name_length: int = MAX_NODE_NAME_LENGTH

    # Security features
    enable_prompt_injection_detection: bool = True
    enable_input_sanitization: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True

    # Rate limiting (requests per minute)
    rate_limit_requests_per_minute: int = 100

    # Allowed file extensions for any file operations
    allowed_file_extensions: set[str] = field(default_factory=lambda: {".json", ".txt", ".md"})


class SecurityError(Exception):
    """Base security exception."""

    pass


class InputValidationError(SecurityError):
    """Raised when input validation fails."""

    pass


class PromptInjectionError(SecurityError):
    """Raised when prompt injection is detected."""

    pass


class RateLimitError(SecurityError):
    """Raised when rate limit is exceeded."""

    pass


class ResourceLimitError(SecurityError):
    """Raised when resource limits are exceeded."""

    pass


class SecurityValidator:
    """Main security validation class."""

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self._request_timestamps: dict[str, list[float]] = {}

    def validate_input_data(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize input data."""
        if not isinstance(input_data, dict):
            security_logger.log_input_validation_failure(
                "input_data", "Input must be a dictionary", input_data
            )
            raise InputValidationError("Input data must be a dictionary")

        validated_data = {}

        for key, value in input_data.items():
            # Validate key format
            if not isinstance(key, str) or len(key) > 100:
                security_logger.log_input_validation_failure(
                    "dict_key", f"Invalid key format: {key}", key
                )
                raise InputValidationError(f"Invalid key format: {key}")

            # Validate and sanitize value
            validated_data[key] = self._validate_value(value, f"input_data.{key}")

        return validated_data

    def validate_message_content(self, content: str, message_role: str = "user") -> str:
        """Validate and sanitize message content."""
        if not isinstance(content, str):
            security_logger.log_input_validation_failure(
                "message_content", "Content must be a string", content
            )
            raise InputValidationError("Message content must be a string")

        # Check length limits
        if len(content) > self.config.max_input_length:
            security_logger.log_input_validation_failure(
                "message_content", f"Content too long: {len(content)} chars", content[:100]
            )
            raise InputValidationError(f"Message content too long: {len(content)} characters")

        # Check for prompt injection if enabled
        if self.config.enable_prompt_injection_detection and message_role == "user":
            self._detect_prompt_injection(content)

        # Sanitize content if enabled
        if self.config.enable_input_sanitization:
            content = self._sanitize_content(content)

        return content

    def validate_thread_id(self, thread_id: str) -> str:
        """Validate thread ID format and security."""
        if not isinstance(thread_id, str):
            security_logger.log_input_validation_failure(
                "thread_id", "Thread ID must be a string", thread_id
            )
            raise InputValidationError("Thread ID must be a string")

        if len(thread_id) > self.config.max_thread_id_length:
            security_logger.log_input_validation_failure(
                "thread_id", f"Thread ID too long: {len(thread_id)}", thread_id
            )
            raise InputValidationError(f"Thread ID too long: {len(thread_id)}")

        if not THREAD_ID_PATTERN.match(thread_id):
            security_logger.log_input_validation_failure(
                "thread_id", "Invalid thread ID format", thread_id
            )
            raise InputValidationError("Thread ID contains invalid characters")

        return thread_id

    def validate_node_name(self, node_name: str) -> str:
        """Validate node name format and security."""
        if not isinstance(node_name, str):
            security_logger.log_input_validation_failure(
                "node_name", "Node name must be a string", node_name
            )
            raise InputValidationError("Node name must be a string")

        if len(node_name) > self.config.max_node_name_length:
            security_logger.log_input_validation_failure(
                "node_name", f"Node name too long: {len(node_name)}", node_name
            )
            raise InputValidationError(f"Node name too long: {len(node_name)}")

        if not NODE_NAME_PATTERN.match(node_name):
            security_logger.log_input_validation_failure(
                "node_name", "Invalid node name format", node_name
            )
            raise InputValidationError("Node name contains invalid characters")

        return node_name

    def validate_config_dict(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            security_logger.log_input_validation_failure(
                "config", "Config must be a dictionary", config
            )
            raise InputValidationError("Config must be a dictionary")

        validated_config = {}

        for key, value in config.items():
            # Validate key format
            if not isinstance(key, str) or not CONFIG_KEY_PATTERN.match(key):
                security_logger.log_input_validation_failure(
                    "config_key", f"Invalid config key: {key}", key
                )
                raise InputValidationError(f"Invalid config key format: {key}")

            # Validate value
            validated_config[key] = self._validate_value(value, f"config.{key}")

        return validated_config

    def check_rate_limit(self, identifier: str) -> None:
        """Check if rate limit is exceeded for the given identifier."""
        if not self.config.enable_rate_limiting:
            return

        current_time = time.time()
        window_start = current_time - 60  # 1-minute window

        # Clean old timestamps
        if identifier in self._request_timestamps:
            self._request_timestamps[identifier] = [
                ts for ts in self._request_timestamps[identifier] if ts > window_start
            ]
        else:
            self._request_timestamps[identifier] = []

        # Check rate limit
        request_count = len(self._request_timestamps[identifier])
        if request_count >= self.config.rate_limit_requests_per_minute:
            security_logger.log_security_event(
                "RATE_LIMIT_EXCEEDED",
                {
                    "identifier": identifier,
                    "request_count": request_count,
                    "limit": self.config.rate_limit_requests_per_minute,
                },
                "WARNING",
            )
            raise RateLimitError(
                f"Rate limit exceeded: {request_count} requests in the last minute"
            )

        # Add current request
        self._request_timestamps[identifier].append(current_time)

    def check_resource_limits(self, context_size: int, execution_step: int) -> None:
        """Check if resource limits are exceeded."""
        if context_size > self.config.max_context_messages:
            security_logger.log_security_event(
                "CONTEXT_SIZE_LIMIT_EXCEEDED",
                {"context_size": context_size, "limit": self.config.max_context_messages},
                "WARNING",
            )
            raise ResourceLimitError(f"Context size limit exceeded: {context_size} messages")

        if execution_step > self.config.max_execution_steps:
            security_logger.log_security_event(
                "EXECUTION_STEPS_LIMIT_EXCEEDED",
                {"execution_step": execution_step, "limit": self.config.max_execution_steps},
                "WARNING",
            )
            raise ResourceLimitError(f"Execution steps limit exceeded: {execution_step} steps")

    def _validate_value(self, value: Any, field_path: str) -> Any:
        """Validate individual values based on type."""
        if isinstance(value, str):
            if len(value) > self.config.max_input_length:
                security_logger.log_input_validation_failure(
                    field_path, f"String too long: {len(value)}", value[:100]
                )
                raise InputValidationError(f"String too long in {field_path}")
            return self._sanitize_content(value) if self.config.enable_input_sanitization else value

        elif isinstance(value, (int, float, bool)):
            return value

        elif isinstance(value, list):
            if len(value) > 1000:  # Limit list size
                security_logger.log_input_validation_failure(
                    field_path, f"List too long: {len(value)}", str(value)[:100]
                )
                raise InputValidationError(f"List too long in {field_path}")
            return [
                self._validate_value(item, f"{field_path}[{i}]") for i, item in enumerate(value)
            ]

        elif isinstance(value, dict):
            if len(value) > 100:  # Limit dict size
                security_logger.log_input_validation_failure(
                    field_path, f"Dict too large: {len(value)}", str(value)[:100]
                )
                raise InputValidationError(f"Dictionary too large in {field_path}")
            return {k: self._validate_value(v, f"{field_path}.{k}") for k, v in value.items()}

        elif value is None:
            return None

        else:
            security_logger.log_input_validation_failure(
                field_path, f"Unsupported type: {type(value)}", str(value)[:100]
            )
            raise InputValidationError(f"Unsupported value type in {field_path}: {type(value)}")

    def _detect_prompt_injection(self, content: str) -> None:
        """Detect potential prompt injection attempts."""
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern.search(content):
                security_logger.log_security_event(
                    "PROMPT_INJECTION_DETECTED",
                    {"pattern": pattern.pattern, "content_preview": content[:200]},
                    "ERROR",
                )
                raise PromptInjectionError(
                    f"Potential prompt injection detected: {pattern.pattern}"
                )

    def _sanitize_content(self, content: str) -> str:
        """Sanitize content by removing potentially dangerous elements."""
        # Remove or escape HTML/JavaScript
        content = re.sub(r"<script.*?</script>", "", content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r"<iframe.*?</iframe>", "", content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r"javascript:", "", content, flags=re.IGNORECASE)

        # Remove control characters except common whitespace
        content = "".join(char for char in content if ord(char) >= 32 or char in "\t\n\r")

        # Limit consecutive whitespace
        content = re.sub(r"\s{10,}", " " * 10, content)

        return content


def safe_json_loads(json_str: str, max_size: int = 10000) -> Any:
    """Safely load JSON with size limits."""
    if len(json_str) > max_size:
        security_logger.log_security_event(
            "JSON_SIZE_LIMIT_EXCEEDED", {"size": len(json_str), "limit": max_size}, "WARNING"
        )
        raise SecurityError(f"JSON string too large: {len(json_str)} bytes")

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        security_logger.log_security_event(
            "JSON_DECODE_ERROR", {"error": str(e), "json_preview": json_str[:200]}, "WARNING"
        )
        raise SecurityError(f"Invalid JSON: {e}")


def require_security_validation(validator: SecurityValidator | None = None):
    """Decorator to enforce security validation on function calls."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get or create validator
            sec_validator = validator or SecurityValidator()

            # Validate arguments if they contain user input
            for arg in args:
                if isinstance(arg, dict) and "messages" in arg:
                    sec_validator.validate_input_data(arg)

            # Execute function
            try:
                result = func(*args, **kwargs)
                security_logger.log_security_event(
                    "FUNCTION_EXECUTED",
                    {"function": func.__name__, "module": func.__module__},
                    "INFO",
                )
                return result
            except Exception as e:
                security_logger.log_security_event(
                    "FUNCTION_EXECUTION_ERROR",
                    {"function": func.__name__, "module": func.__module__, "error": str(e)},
                    "ERROR",
                )
                raise

        return wrapper

    return decorator


# Global security validator instance
default_security_validator = SecurityValidator()
