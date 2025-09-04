#!/usr/bin/env python3
"""Test script to verify optional dependencies work correctly."""

print("Testing optional dependencies...")

# Test basic imports
print("✓ Importing core pyagenity modules...")
from pyagenity.utils import Message
from pyagenity.state import AgentState
from pyagenity.checkpointer import BaseCheckpointer, InMemoryCheckpointer

# Test checkpointer imports
print("✓ Testing checkpointer imports...")
try:
    from pyagenity.checkpointer import PgCheckpointer

    print("✓ PgCheckpointer available (pg_checkpoint dependencies installed)")
except ImportError as e:
    print("⚠️  PgCheckpointer not available (install with: pip install pyagenity[pg_checkpoint])")
    print(f"   Error: {e}")

# Test MCP functionality
print("✓ Testing MCP imports...")
try:
    from pyagenity.graph.tool_node import ToolNode

    # Try to create a ToolNode with MCP client (this should fail gracefully)
    tool_node = ToolNode([])
    print("✓ ToolNode available for basic functionality")

    # Try with MCP client
    try:
        tool_node_with_mcp = ToolNode([], client="dummy")  # This should raise ImportError
    except ImportError as e:
        print("✓ MCP client functionality properly protected")
        print(f"   Expected error: {e}")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")

except ImportError as e:
    print(f"✗ ToolNode import failed: {e}")

# Test publisher imports (should work as they have their own protection)
print("✓ Testing publisher imports...")
try:
    from pyagenity.publisher.redis_publisher import RedisPublisher

    print("✓ RedisPublisher available (with lazy loading)")
except ImportError as e:
    print(f"✗ RedisPublisher import failed: {e}")

print("\n🎉 Optional dependency testing completed!")
