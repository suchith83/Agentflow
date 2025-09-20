"""Constants for ToolNode package.

Separated to avoid circular imports and keep public API tidy.
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
