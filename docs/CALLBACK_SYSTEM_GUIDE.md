# PyAgenity Callback System

The PyAgenity callback system has been successfully implemented to replace the security module with a flexible, user-defined validation and monitoring system.

## Overview

The callback system allows users to define custom validation logic for three types of invocations:
- **AI** calls (LLM model invocations)
- **TOOL** calls (internal tool executions)
- **MCP** calls (Model Context Protocol executions)

## Callback Types

### 1. BeforeInvokeCallback
Called before any invocation occurs. Can modify input data or block execution entirely.

```python
from pyagenity.utils import BeforeInvokeCallback, CallbackContext, InvocationType

class SecurityValidationCallback(BeforeInvokeCallback[dict, dict]):
    async def __call__(self, context: CallbackContext, input_data: dict) -> dict:
        # Validate and potentially modify input data
        if "dangerous_keyword" in str(input_data):
            raise ValueError("Dangerous content detected")
        return input_data
```

### 2. AfterInvokeCallback
Called after successful invocation. Can modify output data or log results.

```python
from pyagenity.utils import AfterInvokeCallback

class AuditLoggingCallback(AfterInvokeCallback[dict, str]):
    async def __call__(self, context: CallbackContext, input_data: dict, output_data: str) -> str:
        # Log the successful invocation
        logger.info(f"Successful {context.invocation_type.value} call to {context.function_name}")
        return output_data
```

### 3. OnErrorCallback
Called when an error occurs during invocation. Can provide error recovery or logging.

```python
from pyagenity.utils import OnErrorCallback

class ErrorHandlingCallback(OnErrorCallback):
    async def __call__(self, context: CallbackContext, input_data: Any, error: Exception) -> Any:
        # Handle errors and optionally provide recovery
        logger.error(f"Error in {context.invocation_type.value}: {error}")
        if context.invocation_type == InvocationType.AI:
            return "Error occurred, using fallback response"
        return None  # No recovery
```

## Usage

### Global Registration (affects all graphs)
```python
from pyagenity.utils import register_before_invoke, register_after_invoke, register_on_error, InvocationType

# Register callbacks globally
register_before_invoke(InvocationType.AI, SecurityValidationCallback())
register_after_invoke(InvocationType.AI, AuditLoggingCallback())
register_on_error(InvocationType.AI, ErrorHandlingCallback())
```

### Per-Graph Registration
```python
from pyagenity.utils import CallbackManager
from pyagenity.graph import StateGraph

# Create a custom callback manager
callback_manager = CallbackManager()
callback_manager.register_before_invoke(InvocationType.AI, SecurityValidationCallback())
callback_manager.register_after_invoke(InvocationType.AI, AuditLoggingCallback())

# Use with a specific graph
graph = StateGraph(AgentState)
# ... add nodes and edges ...
compiled = graph.compile(callback_manager=callback_manager)
```

## Integration Points

The callback system is integrated at the following points in the execution flow:

1. **Node.execute()** - AI function calls
2. **ToolNode._internal_execute()** - Internal tool calls
3. **ToolNode._mcp_execute()** - MCP tool calls

## Features

- ✅ **Type-safe** with modern Python generics
- ✅ **Async support** for all callback types
- ✅ **Flexible validation** - callbacks can modify input/output or block execution
- ✅ **Error recovery** - error callbacks can provide fallback responses
- ✅ **Invocation-specific** - different callbacks for AI, TOOL, and MCP calls
- ✅ **Global and per-graph** configuration options
- ✅ **Performance optimized** - minimal overhead when no callbacks are registered

## Migration from Security Module

The old security module has been completely replaced. Where you previously had:

```python
# OLD - security module approach
from pyagenity.utils import SecurityValidator
security = SecurityValidator()
security.validate_input(input_data)
```

You now use:

```python
# NEW - callback system approach
class CustomValidationCallback(BeforeInvokeCallback[dict, dict]):
    async def __call__(self, context: CallbackContext, input_data: dict) -> dict:
        # Your validation logic here
        return input_data

register_before_invoke(InvocationType.AI, CustomValidationCallback())
```

## Examples

See `examples/callback-validation/validation_examples.py` for comprehensive examples of:
- Security validation
- Content filtering
- Audit logging
- Error handling and recovery
- Rate limiting
- Circuit breaker patterns

The callback system provides a powerful, flexible foundation for implementing any validation, monitoring, or modification logic needed for your PyAgenity applications.
