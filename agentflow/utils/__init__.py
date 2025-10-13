"""
Unified utility exports for TAF agent graphs.

This module re-exports core utility symbols for agent graph construction, message handling,
callback management, reducers, and constants. Import from this module for a stable, unified
surface of agent utilities.

Main Exports:
    - Message and content blocks (Message, TextBlock, ToolCallBlock, etc.)
    - Callback management (CallbackManager, register_before_invoke, etc.)
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
    BeforeInvokeCallback,
    CallbackContext,
    CallbackManager,
    InvocationType,
    OnErrorCallback,
    default_callback_manager,
    register_after_invoke,
    register_before_invoke,
    register_on_error,
)
from .command import Command

# Export from constants.py
from .constants import END, START, ExecutionState, ResponseGranularity, StorageLevel
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
from .thread_info import ThreadInfo
from .thread_name_generator import generate_dummy_thread_name


__all__ = [
    "END",
    "START",
    "AfterInvokeCallback",
    "AsyncIDGenerator",
    "BackgroundTaskManager",
    "BaseIDGenerator",
    "BeforeInvokeCallback",
    "BigIntIDGenerator",
    "CallbackContext",
    "CallbackManager",
    "Command",
    "DefaultIDGenerator",
    "ExecutionState",
    "HexIDGenerator",
    "IDType",
    "IntIDGenerator",
    "InvocationType",
    "OnErrorCallback",
    "ResponseGranularity",
    "ShortIDGenerator",
    "StorageLevel",
    "TaskMetadata",
    "ThreadInfo",
    "TimestampIDGenerator",
    "UUIDGenerator",
    "add_messages",
    "append_items",
    "call_sync_or_async",
    "configure_logging",
    "convert_messages",
    "default_callback_manager",
    "generate_dummy_thread_name",
    "register_after_invoke",
    "register_before_invoke",
    "register_on_error",
    "replace_messages",
    "replace_value",
    "run_coroutine",
]
