"""
Callback system for PyAgenity.

This module provides a comprehensive callback framework that allows users to define
their own validation logic and custom behavior at key points in the execution flow:

- before_invoke: Called before AI/TOOL/MCP invocation for input validation and modification
- after_invoke: Called after AI/TOOL/MCP invocation for output validation and modification
- on_error: Called when errors occur during invocation for error handling and logging

The system is generic and type-safe, supporting different callback types for different
invocation contexts.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union


logger = logging.getLogger(__name__)


class InvocationType(Enum):
    """Types of invocations that can trigger callbacks."""

    AI = "ai"
    TOOL = "tool"
    MCP = "mcp"


@dataclass
class CallbackContext:
    """Context information passed to callbacks."""

    invocation_type: InvocationType
    node_name: str
    function_name: str | None = None
    metadata: dict[str, Any] | None = None


class BeforeInvokeCallback[T, R](ABC):
    """Abstract base class for before_invoke callbacks.

    Called before the AI model, tool, or MCP function is invoked.
    Allows for input validation and modification.
    """

    @abstractmethod
    async def __call__(self, context: CallbackContext, input_data: T) -> Union[T, R]:
        """Execute the before_invoke callback.

        Args:
            context: Context information about the invocation
            input_data: The input data about to be sent to the invocation

        Returns:
            Modified input data (can be same type or different type)

        Raises:
            Exception: If validation fails or modification cannot be performed
        """
        ...


class AfterInvokeCallback[T, R](ABC):
    """Abstract base class for after_invoke callbacks.

    Called after the AI model, tool, or MCP function is invoked.
    Allows for output validation and modification.
    """

    @abstractmethod
    async def __call__(
        self, context: CallbackContext, input_data: T, output_data: Any
    ) -> Union[Any, R]:
        """Execute the after_invoke callback.

        Args:
            context: Context information about the invocation
            input_data: The original input data that was sent
            output_data: The output data returned from the invocation

        Returns:
            Modified output data (can be same type or different type)

        Raises:
            Exception: If validation fails or modification cannot be performed
        """
        ...


class OnErrorCallback(ABC):
    """Abstract base class for on_error callbacks.

    Called when an error occurs during invocation.
    Allows for error handling and logging.
    """

    @abstractmethod
    async def __call__(
        self, context: CallbackContext, input_data: Any, error: Exception
    ) -> Any | None:
        """Execute the on_error callback.

        Args:
            context: Context information about the invocation
            input_data: The input data that caused the error
            error: The exception that occurred

        Returns:
            Optional recovery value or None to re-raise the error

        Raises:
            Exception: If error handling fails or if the error should be re-raised
        """
        ...


# Type aliases for cleaner type hints
BeforeInvokeCallbackType = Union[
    BeforeInvokeCallback[Any, Any], Callable[[CallbackContext, Any], Union[Any, Awaitable[Any]]]
]

AfterInvokeCallbackType = Union[
    AfterInvokeCallback[Any, Any], Callable[[CallbackContext, Any, Any], Union[Any, Awaitable[Any]]]
]

OnErrorCallbackType = Union[
    OnErrorCallback,
    Callable[[CallbackContext, Any, Exception], Union[Any | None, Awaitable[Any | None]]],
]


class CallbackManager:
    """Manages registration and execution of callbacks for different invocation types."""

    def __init__(self):
        self._before_callbacks: dict[InvocationType, list[BeforeInvokeCallbackType]] = {
            InvocationType.AI: [],
            InvocationType.TOOL: [],
            InvocationType.MCP: [],
        }
        self._after_callbacks: dict[InvocationType, list[AfterInvokeCallbackType]] = {
            InvocationType.AI: [],
            InvocationType.TOOL: [],
            InvocationType.MCP: [],
        }
        self._error_callbacks: dict[InvocationType, list[OnErrorCallbackType]] = {
            InvocationType.AI: [],
            InvocationType.TOOL: [],
            InvocationType.MCP: [],
        }

    def register_before_invoke(
        self, invocation_type: InvocationType, callback: BeforeInvokeCallbackType
    ) -> None:
        """Register a before_invoke callback for a specific invocation type."""
        self._before_callbacks[invocation_type].append(callback)

    def register_after_invoke(
        self, invocation_type: InvocationType, callback: AfterInvokeCallbackType
    ) -> None:
        """Register an after_invoke callback for a specific invocation type."""
        self._after_callbacks[invocation_type].append(callback)

    def register_on_error(
        self, invocation_type: InvocationType, callback: OnErrorCallbackType
    ) -> None:
        """Register an on_error callback for a specific invocation type."""
        self._error_callbacks[invocation_type].append(callback)

    async def execute_before_invoke(self, context: CallbackContext, input_data: Any) -> Any:
        """Execute all before_invoke callbacks for the given context."""
        current_data = input_data

        for callback in self._before_callbacks[context.invocation_type]:
            try:
                if isinstance(callback, BeforeInvokeCallback):
                    current_data = await callback(context, current_data)
                elif callable(callback):
                    # Handle both sync and async callables
                    result = callback(context, current_data)
                    if hasattr(result, "__await__"):
                        current_data = await result
                    else:
                        current_data = result
            except Exception as e:
                # If before_invoke callback fails, trigger error callbacks
                await self.execute_on_error(context, input_data, e)
                raise

        return current_data

    async def execute_after_invoke(
        self, context: CallbackContext, input_data: Any, output_data: Any
    ) -> Any:
        """Execute all after_invoke callbacks for the given context."""
        current_output = output_data

        for callback in self._after_callbacks[context.invocation_type]:
            try:
                if isinstance(callback, AfterInvokeCallback):
                    current_output = await callback(context, input_data, current_output)
                elif callable(callback):
                    # Handle both sync and async callables
                    result = callback(context, input_data, current_output)
                    if hasattr(result, "__await__"):
                        current_output = await result
                    else:
                        current_output = result
            except Exception as e:
                # If after_invoke callback fails, trigger error callbacks
                await self.execute_on_error(context, input_data, e)
                raise

        return current_output

    async def execute_on_error(
        self, context: CallbackContext, input_data: Any, error: Exception
    ) -> Any | None:
        """Execute all on_error callbacks for the given context."""
        recovery_value = None

        for callback in self._error_callbacks[context.invocation_type]:
            try:
                result = None
                if isinstance(callback, OnErrorCallback):
                    result = await callback(context, input_data, error)
                elif callable(callback):
                    # Handle both sync and async callables
                    result = callback(context, input_data, error)
                    if hasattr(result, "__await__"):
                        result = await result  # type: ignore

                # If any error callback returns a value, use it as recovery
                if result is not None:
                    recovery_value = result
            except Exception as exc:
                # If error callback itself fails, continue with other callbacks
                # but don't let it break the error handling flow
                logger.exception("Error callback failed: %s", exc)
                continue

        return recovery_value

    def clear_callbacks(self, invocation_type: InvocationType | None = None) -> None:
        """Clear callbacks for a specific invocation type or all types."""
        if invocation_type:
            self._before_callbacks[invocation_type].clear()
            self._after_callbacks[invocation_type].clear()
            self._error_callbacks[invocation_type].clear()
        else:
            for inv_type in InvocationType:
                self._before_callbacks[inv_type].clear()
                self._after_callbacks[inv_type].clear()
                self._error_callbacks[inv_type].clear()

    def get_callback_counts(self) -> dict[str, dict[str, int]]:
        """Get count of registered callbacks by type for debugging."""
        return {
            inv_type.value: {
                "before_invoke": len(self._before_callbacks[inv_type]),
                "after_invoke": len(self._after_callbacks[inv_type]),
                "on_error": len(self._error_callbacks[inv_type]),
            }
            for inv_type in InvocationType
        }


# Global default callback manager instance
default_callback_manager = CallbackManager()


# Convenience functions for the global callback manager
def register_before_invoke(
    invocation_type: InvocationType, callback: BeforeInvokeCallbackType
) -> None:
    """Register a before_invoke callback on the global callback manager."""
    default_callback_manager.register_before_invoke(invocation_type, callback)


def register_after_invoke(
    invocation_type: InvocationType, callback: AfterInvokeCallbackType
) -> None:
    """Register an after_invoke callback on the global callback manager."""
    default_callback_manager.register_after_invoke(invocation_type, callback)


def register_on_error(invocation_type: InvocationType, callback: OnErrorCallbackType) -> None:
    """Register an on_error callback on the global callback manager."""
    default_callback_manager.register_on_error(invocation_type, callback)
