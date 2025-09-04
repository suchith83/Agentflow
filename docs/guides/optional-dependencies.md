# Optional Dependencies Guide

PyAgenity uses optional dependencies to keep the core installation lightweight while providing powerful extensions when needed.

## Available Extras

### 1. PostgreSQL + Redis Checkpointing (`pg_checkpoint`)

**Install:** `pip install pyagenity[pg_checkpoint]`

**Includes:**
- `asyncpg>=0.29.0` - PostgreSQL async driver
- `redis>=4.2` - Redis client

**Provides:**
- `PgCheckpointer` class for persistent state storage
- PostgreSQL database schema management
- Redis caching layer for performance
- Connection pooling and retry logic

**Usage:**
```python
from pyagenity.checkpointer import PgCheckpointer

# This will work only with pg_checkpoint extra installed
checkpointer = PgCheckpointer(
    postgres_dsn="postgresql://user:pass@localhost/db",
    redis_url="redis://localhost:6379/0"
)
```

### 2. Model Context Protocol (`mcp`)

**Install:** `pip install pyagenity[mcp]`

**Includes:**
- `fastmcp>=2.11.3` - Fast MCP client implementation
- `mcp>=1.13.0` - MCP protocol definitions

**Provides:**
- MCP client support in `ToolNode`
- Integration with MCP servers for external tools
- Remote tool execution capabilities

**Usage:**
```python
from pyagenity.graph.tool_node import ToolNode
from fastmcp import Client

# MCP client requires the mcp extra
client = Client("path/to/mcp/server")
tool_node = ToolNode([], client=client)  # This will fail without mcp extra
```

### 3. Publisher Integrations

#### Redis Publisher (`redis`)
**Install:** `pip install pyagenity[redis]`
- Uses lazy loading - will work without extra but fail at runtime if redis not installed

#### Kafka Publisher (`kafka`)  
**Install:** `pip install pyagenity[kafka]`
- `aiokafka>=0.8.0`

#### RabbitMQ Publisher (`rabbitmq`)
**Install:** `pip install pyagenity[rabbitmq]`  
- `aio-pika>=9.0.0`

## Multiple Extras

You can install multiple extras together:

```bash
# PostgreSQL checkpointing + MCP support
pip install pyagenity[pg_checkpoint,mcp]

# All publishers
pip install pyagenity[redis,kafka,rabbitmq]

# Everything
pip install pyagenity[pg_checkpoint,mcp,redis,kafka,rabbitmq]
```

## Base Installation

The base installation includes only essential dependencies:

```bash
pip install pyagenity
```

**Includes:**
- `injectq>=0.1.0` - Dependency injection
- `litellm>=1.75.0` - LLM interface  
- `pydantic>=2.0.0` - Data validation
- `python-dotenv>=1.0.0` - Environment configuration

**Available without extras:**
- Core agents and state management
- `InMemoryCheckpointer` 
- `BaseCheckpointer` interface
- Graph execution engine
- Basic `ToolNode` functionality (no MCP)
- All utility functions

## Error Messages

When optional dependencies are missing, you'll see helpful error messages:

```python
# Without pg_checkpoint extra
from pyagenity.checkpointer import PgCheckpointer
# ImportError: PgCheckpointer requires 'asyncpg' package. 
# Install with: pip install pyagenity[pg_checkpoint]

# Without mcp extra  
tool_node = ToolNode([], client=mcp_client)
# ImportError: MCP client functionality requires 'fastmcp' and 'mcp' packages.
# Install with: pip install pyagenity[mcp]
```

## Development

For development, install all extras:

```bash
pip install pyagenity[pg_checkpoint,mcp,redis,kafka,rabbitmq]
```

This ensures you can test all functionality locally.
