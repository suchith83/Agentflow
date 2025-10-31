"""
Unified utility exports for TAF agent graphs.

This module re-exports core utility symbols for agent graph construction, message handling,
callback management, reducers, and constants. Import from this module for a stable, unified
surface of agent utilities.

Main Exports:
    - Message and content blocks (Message, TextBlock, ToolCallBlock, etc.)
    - Callback management (CallbackManager, register_before_invoke, etc.)
    - Validators (PromptInjectionValidator, MessageContentValidator, etc.)
    - Command and callable utilities (Command, call_sync_or_async)
    - Reducers (add_messages, replace_messages, append_items, replace_value)
    - Constants (START, END, ExecutionState, etc.)
    - Converter (convert_messages)
"""

from agentflow.state.reducers import add_messages, append_items, replace_messages, replace_value

from .background_task_manager import BackgroundTaskManager, TaskMetadata
from .callable_utils import call_sync_or_async, run_coroutine

# Export from callbacks.py
from .callbacks import (
    AfterInvokeCallback,
    BaseValidator,
    BeforeInvokeCallback,
    CallbackContext,
    CallbackManager,
    InvocationType,
    OnErrorCallback,
)
from .command import Command

# Export from constants.py
from .constants import END, START, ExecutionState, ResponseGranularity
from .converter import convert_messages
from .id_generator import (
    AsyncIDGenerator,
    BaseIDGenerator,
    BigIntIDGenerator,
    DefaultIDGenerator,
    HexIDGenerator,
    IDType,
    IntIDGenerator,
    ShortIDGenerator,
    TimestampIDGenerator,
    UUIDGenerator,
)
from .logging import configure_logging
from .shutdown import (
    DelayedKeyboardInterrupt,
    GracefulShutdownManager,
    delayed_keyboard_interrupt,
    setup_exception_handler,
    shutdown_with_timeout,
)
from .thread_info import ThreadInfo

# Export validators
from .validators import (
    MessageContentValidator,
    PromptInjectionValidator,
    ValidationError,
    register_default_validators,
)


__all__ = [
    "END",
    "START",
    "AfterInvokeCallback",
    "AsyncIDGenerator",
    "BackgroundTaskManager",
    "BaseIDGenerator",
    "BaseValidator",
    "BeforeInvokeCallback",
    "BigIntIDGenerator",
    "CallbackContext",
    "CallbackManager",
    "Command",
    "DefaultIDGenerator",
    "DelayedKeyboardInterrupt",
    "ExecutionState",
    "GracefulShutdownManager",
    "HexIDGenerator",
    "IDType",
    "IntIDGenerator",
    "InvocationType",
    "MessageContentValidator",
    "OnErrorCallback",
    "PromptInjectionValidator",
    "ResponseGranularity",
    "ShortIDGenerator",
    "StorageLevel",
    "TaskMetadata",
    "ThreadInfo",
    "TimestampIDGenerator",
    "UUIDGenerator",
    "ValidationError",
    "add_messages",
    "append_items",
    "call_sync_or_async",
    "configure_logging",
    "convert_messages",
    "delayed_keyboard_interrupt",
    "register_default_validators",
    "replace_messages",
    "replace_value",
    "run_coroutine",
    "setup_exception_handler",
    "shutdown_with_timeout",
]
