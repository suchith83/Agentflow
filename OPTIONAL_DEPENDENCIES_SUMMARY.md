# Optional Dependencies Implementation Summary

## Overview

Successfully implemented optional dependencies for PyAgenity to keep the core installation lightweight while providing powerful extensions when needed.

## Changes Made

### 1. Updated `pyproject.toml`

**Moved MCP dependencies from core to optional:**
- Removed `fastmcp>=2.11.3` and `mcp>=1.13.0` from main dependencies
- Created new optional dependency groups:

```toml
[project.optional-dependencies]
pg_checkpoint = [
    "asyncpg>=0.29.0",
    "redis>=4.2"
]
mcp = [
    "fastmcp>=2.11.3",
    "mcp>=1.13.0"
]
composio = [
    "composio>=0.8.0"
]
langchain = [
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "tavily-python>=0.5.0"
]
redis = ["redis>=4.2"]
kafka = ["aiokafka>=0.8.0"]
rabbitmq = ["aio-pika>=9.0.0"]
```

### 2. Updated `pg_checkpointer.py`

**Added optional dependency handling:**
- Wrapped asyncpg and redis imports in try/except blocks
- Added `HAS_ASYNCPG` and `HAS_REDIS` flags
- Added dependency checks in `__init__` with helpful error messages
- Updated type annotations to use `Any` when dependencies might not be available
- Fixed retry logic to handle missing asyncpg exception types

**Key changes:**
```python
try:
    import asyncpg
    from asyncpg import Pool
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None
```

### 3. Updated `tool_node.py`

**Added MCP optional dependency handling:**
- Wrapped fastmcp and mcp imports in try/except blocks
- Added `HAS_FASTMCP` and `HAS_MCP` flags
- Added dependency checks when MCP client is provided
- Updated type annotations to use `Any` when MCP dependencies might not be available

**Key changes:**
```python
if client is not None:
    if not HAS_FASTMCP or not HAS_MCP:
        raise ImportError(
            "MCP client functionality requires 'fastmcp' and 'mcp' packages. "
            "Install with: pip install pyagenity[mcp]"
        )
```

**Added Composio optional dependency handling and adapter integration:**
- New integrations package at `pyagenity/integrations/` with `composio_adapter.py`
- `ToolNode` constructor accepts `composio_adapter` and can discover/execute Composio tools
- Discovery via `list_raw_tools_for_llm` to expose tools as function-calling specs
- Execution path `_composio_execute` mirrors MCP/local flows with callbacks and events

Install the extra to enable:

```bash
pip install pyagenity[composio]
```

**Added LangChain optional dependency handling and adapter integration:**
- New adapter at `pyagenity/adapters/tools/langchain_adapter.py` with a selective set (Tavily, Requests)
- `ToolNode` constructor accepts `langchain_adapter` and can discover/execute LangChain tools
- Discovery via `list_tools_for_llm` to expose tools as function-calling specs
- Execution path `_langchain_execute` mirrors other flows with callbacks and events

Install the extra to enable:

```bash
pip install pyagenity[langchain]
```

### 4. Updated `checkpointer/__init__.py`

**Conditional PgCheckpointer export:**
- Made PgCheckpointer import conditional
- Only export PgCheckpointer if dependencies are available
- Graceful fallback if asyncpg/redis not installed

### 5. Documentation Updates

**Updated README.md:**
- Added optional dependencies installation section
- Documented all available extras with examples
- Explained how to install multiple extras

**Created `docs/guides/optional-dependencies.md`:**
- Comprehensive guide to all optional dependencies
- Usage examples and error message explanations
- Development setup instructions

**Created example demo:**
- `examples/optional-dependencies/demo.py` - shows how optional deps work
- Demonstrates graceful degradation when deps not available

## Installation Options

### Base Installation (Minimal)
```bash
pip install pyagenity
```
**Includes:** injectq, litellm, pydantic, python-dotenv
**Available:** Core agents, InMemoryCheckpointer, basic ToolNode, utilities

### With PostgreSQL + Redis Checkpointing
```bash
pip install pyagenity[pg_checkpoint]
```
**Adds:** asyncpg, redis
**Enables:** PgCheckpointer with database persistence and caching

### With MCP Support
```bash
pip install pyagenity[mcp]
```
**Adds:** fastmcp, mcp
**Enables:** MCP client integration in ToolNode

### With Composio Tools
```bash
pip install pyagenity[composio]
```
**Adds:** composio
**Enables:** Composio adapter integration in ToolNode for tool discovery and execution

### With LangChain Tools (registry-based)
```bash
pip install pyagenity[langchain]
```
**Adds:** langchain-core, langchain-community, tavily-python, langchain-tavily
**Enables:** LangChain adapter integration in ToolNode. Register any LangChain tools you like, or rely on the default autoload of a couple of common tools.

Basic usage:

```python
from pyagenity.adapters.tools.langchain_adapter import LangChainAdapter
from pyagenity.graph import ToolNode

# Create adapter and register arbitrary LangChain tools
adapter = LangChainAdapter()  # autoloads a couple defaults if registry empty

# You can also register your own tools explicitly
# from langchain_community.tools import DuckDuckGoSearchRun
# adapter.register_tool(DuckDuckGoSearchRun())

tool_node = ToolNode([], langchain_adapter=adapter)
tools_for_llm = tool_node.all_tools_sync()  # returns unified function-calling schemas
```

To disable the convenience autoload:

```python
adapter = LangChainAdapter(autoload_default_tools=False)
adapter.register_tools([your_tool, another_tool])
```

### Multiple Extras
```bash
pip install pyagenity[pg_checkpoint,mcp,composio,langchain,redis]
```

## Error Messages

Users get helpful error messages when trying to use functionality without required dependencies:

```python
# Without pg_checkpoint extra
from pyagenity.checkpointer import PgCheckpointer
# ImportError: PgCheckpointer requires 'asyncpg' package.
# Install with: pip install pyagenity[pg_checkpoint]

# Without mcp extra
tool_node = ToolNode([], client=mcp_client)
# ImportError: MCP client functionality requires 'fastmcp' and 'mcp' packages.
# Install with: pip install pyagenity[mcp]

# Without langchain extra
# When constructing the adapter directly
from pyagenity.adapters.tools.langchain_adapter import LangChainAdapter
# ImportError: LangChainAdapter requires 'langchain-core' and optional integrations.
# Install with: pip install pyagenity[langchain]
```

## Testing

- ✅ All existing tests continue to pass
- ✅ Basic functionality works without any extras
- ✅ PgCheckpointer works when pg_checkpoint extra installed
- ✅ ToolNode works with and without MCP client
- ✅ Publishers work with lazy loading
- ✅ Helpful error messages when dependencies missing

## Backwards Compatibility

- ✅ All existing code continues to work if dependencies are installed
- ✅ Graceful degradation when dependencies not available
- ✅ Clear error messages guide users to install correct extras
- ✅ No breaking changes to existing APIs

## Benefits

1. **Smaller base installation** - Core functionality has minimal dependencies
2. **Modular architecture** - Install only what you need
3. **Better dependency management** - Clear separation of concerns
4. **Helpful error messages** - Users know exactly what to install
5. **Backwards compatible** - Existing code continues to work
6. **Future extensible** - Easy to add new optional dependency groups

The implementation successfully achieves the goal of making PyAgenity modular while maintaining full functionality for users who install the appropriate extras.
