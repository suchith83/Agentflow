# PyAgenity Logging Implementation

## Overview

I have successfully implemented comprehensive logging throughout the PyAgenity project. The logging system provides:

- **Centralized Configuration**: Located in `pyagenity/utils/logging.py`
- **Module-Specific Loggers**: Each module uses `logging.getLogger(__name__)` for proper hierarchical logging
- **Comprehensive Coverage**: Logging added to all major modules including graph, state, utils, checkpointer, exceptions, and store modules
- **Consistent Formatting**: Timestamped logs with module names for easy debugging

## Usage

### Basic Setup

```python
# Import and configure logging
from pyagenity.utils.logging import configure_logging
import logging

# Configure logging level (DEBUG, INFO, WARNING, ERROR)
configure_logging(level=logging.INFO)
```

### In Your Modules

Each PyAgenity module now includes logging using the standard pattern:

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.info("Function started")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")
```

### Configuration Options

The `configure_logging()` function accepts these parameters:

- `level`: Logging level (default: INFO)
- `format_string`: Custom format (optional)
- `handler`: Custom handler (optional)

### Example Usage

```python
import logging
from pyagenity.utils.logging import configure_logging

# Set to DEBUG to see all logs
configure_logging(level=logging.DEBUG)

# Your PyAgenity code will now log appropriately
from pyagenity.graph import StateGraph

graph = StateGraph()  # This will log "Initializing StateGraph"
```

## Modules with Logging

### Graph Module
- `compiled_graph.py`: Comprehensive logging for graph execution, state management, and streaming
- `node.py`: Logging for node execution, tool calls, and error handling
- `edge.py`: Logging for edge creation
- `state_graph.py`: Logging for graph construction, compilation, and validation
- `tool_node.py`: Logging for tool registration and execution

### State Module
- `agent_state.py`: Logging for state transitions and operations
- `execution_state.py`: Logging for execution status changes and metadata

### Utils Module
- `dependency_injection.py`: Logging for dependency registration and retrieval
- `callbacks.py`: Already had logging implemented

### Checkpointer Module
- `base_checkpointer.py`: Base logging setup
- `in_memory_checkpointer.py`: Logging for state storage and retrieval operations

### Exceptions Module
- `graph_error.py`: Logging when graph errors are raised
- `node_error.py`: Logging when node errors are raised  
- `recursion_error.py`: Logging when recursion errors are raised

### Store Module
- `base_store.py`: Base logging setup for store operations

## Log Output Format

The default log format is:
```
[2025-08-23 19:58:31,261] INFO     pyagenity.graph.compiled_graph: Message here
```

This includes:
- Timestamp
- Log level
- Full module name (for easy tracing)
- Log message

## Benefits

1. **Easy Debugging**: Trace execution flow across modules
2. **Performance Monitoring**: Log timing and resource usage
3. **Error Tracking**: Comprehensive error logging with context
4. **Development Aid**: Debug information for development
5. **Production Monitoring**: Configurable log levels for production use

## Testing

The logging implementation has been tested with:
- Module-specific logger functionality
- Hierarchical logging behavior
- Different log levels
- Proper formatting and timestamps

Run the test with:
```bash
python test_comprehensive_logging.py
```

## Configuration Best Practices

- Use `INFO` level for production
- Use `DEBUG` level for development/debugging  
- Use `WARNING` for important but non-critical issues
- Use `ERROR` for actual errors that need attention
- Configure appropriate handlers for your deployment environment
