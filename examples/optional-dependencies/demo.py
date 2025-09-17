"""Example demonstrating optional dependencies in PyAgenity.

This example shows how different functionality is available based on
which optional dependencies are installed.
"""


def test_basic_functionality():
    """Test that basic PyAgenity works without any optional dependencies."""
    print("🔧 Testing basic functionality...")

    # These should always work
    from pyagenity.checkpointer import InMemoryCheckpointer
    from pyagenity.graph.tool_node import ToolNode
    from pyagenity.state import AgentState
    from pyagenity.utils import Message

    # Create basic objects
    state = AgentState()
    message = Message.from_text("Hello, world!")
    checkpointer = InMemoryCheckpointer()
    tool_node = ToolNode([])

    print("✅ Basic functionality works!")
    return True


def test_pg_checkpoint():
    """Test PostgreSQL + Redis checkpointing functionality."""
    print("\n📊 Testing PostgreSQL checkpointing...")

    try:
        from pyagenity.checkpointer import PgCheckpointer

        print("✅ PgCheckpointer available - pg_checkpoint extra is installed")

        # This would fail if asyncpg/redis weren't available
        try:
            checkpointer = PgCheckpointer(
                postgres_dsn="postgresql://test:test@localhost/test",
                redis_url="redis://localhost:6379/0",
            )
            print("✅ PgCheckpointer can be instantiated")
        except Exception as e:
            print(f"⚠️  PgCheckpointer instantiation failed (database not available): {e}")

        return True

    except ImportError as e:
        print(f"❌ PgCheckpointer not available: {e}")
        print("   Install with: pip install pyagenity[pg_checkpoint]")
        return False


def test_mcp_functionality():
    """Test MCP (Model Context Protocol) functionality."""
    print("\n🔌 Testing MCP functionality...")

    try:
        from pyagenity.graph.tool_node import ToolNode

        # Test basic ToolNode (should work)
        basic_tool_node = ToolNode([])
        print("✅ Basic ToolNode works")

        # Test MCP client integration
        # Check if MCP dependencies are available by importing the flags from tool_node
        try:
            from pyagenity.graph.tool_node import HAS_FASTMCP, HAS_MCP

            mcp_available = HAS_FASTMCP and HAS_MCP
        except ImportError:
            mcp_available = False

        if not mcp_available:
            try:
                # This should fail if MCP dependencies aren't installed
                mcp_tool_node = ToolNode([], client="dummy_client")
                print("❌ This shouldn't happen - MCP client should have been rejected")
                return False

            except ImportError as e:
                print(f"✅ MCP client properly protected: {e}")
                print("   Install with: pip install pyagenity[mcp]")
                return True
        else:
            # Dependencies are installed, so client will be accepted
            try:
                # Create MCP-enabled ToolNode
                mcp_tool_node = ToolNode([], client="dummy_client")
                print("✅ MCP dependencies available - client accepted")
                print("   (Would need real MCP client for actual functionality)")
                return True
            except Exception as e:
                print(f"⚠️  Unexpected error with MCP client: {e}")
                return False

    except ImportError as e:
        print(f"❌ ToolNode import failed: {e}")
        return False


def test_publisher_functionality():
    """Test publisher functionality with lazy loading."""
    print("\n📡 Testing publisher functionality...")

    try:
        from pyagenity.publisher.redis_publisher import RedisPublisher

        print("✅ RedisPublisher import works (uses lazy loading)")

        # The actual Redis connection would fail at runtime if redis isn't installed
        # but the class can be imported
        publisher = RedisPublisher({"url": "redis://localhost:6379"})
        print("✅ RedisPublisher can be instantiated (connection happens lazily)")

        return True

    except ImportError as e:
        print(f"❌ RedisPublisher import failed: {e}")
        return False


def main():
    """Run all optional dependency tests."""
    print("🧪 PyAgenity Optional Dependencies Test\n")
    print("=" * 50)

    results = {}
    results["basic"] = test_basic_functionality()
    results["pg_checkpoint"] = test_pg_checkpoint()
    results["mcp"] = test_mcp_functionality()
    results["publishers"] = test_publisher_functionality()

    print("\n" + "=" * 50)
    print("📋 Test Summary:")

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name:<15}: {status}")

    total_passed = sum(results.values())
    print(f"\n🎯 Result: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("🎉 All functionality working correctly!")
    else:
        print("💡 Some optional features not available - install extras as needed")


if __name__ == "__main__":
    main()
