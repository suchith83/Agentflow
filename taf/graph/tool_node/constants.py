"""Constants for ToolNode package.

This module defines constants used throughout the ToolNode implementation,
particularly parameter names that are automatically injected by the TAF
framework during tool execution. These parameters are excluded from tool
schema generation since they are provided by the execution context.

The constants are separated into their own module to avoid circular imports
and maintain a clean public API.

Parameter names that are automatically injected during tool execution.

These parameters are provided by the TAF framework and should be excluded
from tool schema generation. They represent execution context and framework
services that are available to tool functions but not provided by the user.

Parameters:
    tool_call_id: Unique identifier for the current tool execution.
    state: Current AgentState instance for context-aware execution.
    config: Configuration dictionary with execution settings.
    generated_id: Framework-generated identifier for various purposes.
    context_manager: BaseContextManager instance for cross-node operations.
    publisher: BasePublisher instance for event publishing.
    checkpointer: BaseCheckpointer instance for state persistence.
    store: BaseStore instance for data storage operations.

Note:
    Tool functions can declare these parameters in their signatures to receive
    the corresponding services, but they should not be included in the tool
    schema since they're not user-provided arguments.
"""

INJECTABLE_PARAMS = {
    "tool_call_id",
    "state",
    "config",
    "generated_id",
    "context_manager",
    "publisher",
    "checkpointer",
    "store",
}
