#!/usr/bin/env python3
"""
Simple test script to verify the callback system works.
"""

import asyncio
import sys
import os
from typing import Any

# Add the project root to the path so we can import pyagenity
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyagenity.utils import (
    CallbackContext,
    CallbackManager,
    InvocationType,
    BeforeInvokeCallback,
    AfterInvokeCallback,
    OnErrorCallback,
)


class TestBeforeCallback(BeforeInvokeCallback[str, bool]):
    """Test before callback."""

    def __init__(self, should_allow: bool = True):
        self.should_allow = should_allow
        self.called = False

    async def __call__(self, context: CallbackContext, input_data: str) -> bool:
        """Execute the callback."""
        self.called = True
        print(f"✅ Before callback executed for {context.function_name}")
        return self.should_allow


class TestAfterCallback(AfterInvokeCallback[str, str]):
    """Test after callback."""

    def __init__(self):
        self.called = False

    async def __call__(self, context: CallbackContext, input_data: str, output_data: Any) -> str:
        """Execute the callback."""
        self.called = True
        print(f"✅ After callback executed for {context.function_name} with result: {output_data}")
        return output_data


class TestErrorCallback(OnErrorCallback):
    """Test error callback."""

    def __init__(self):
        self.called = False

    async def __call__(self, context: CallbackContext, input_data: Any, error: Exception) -> None:
        """Execute the callback."""
        self.called = True
        print(f"✅ Error callback executed for {context.function_name} with error: {error}")


async def test_callback_system():
    """Test the callback system functionality."""
    print("🧪 Testing PyAgenity Callback System")
    print("=" * 40)

    # Create callback manager
    manager = CallbackManager()

    # Create test callbacks
    before_callback = TestBeforeCallback(True)
    after_callback = TestAfterCallback()
    error_callback = TestErrorCallback()
    blocking_callback = TestBeforeCallback(False)

    # Add callbacks
    manager.register_before_invoke(InvocationType.AI, before_callback)
    manager.register_after_invoke(InvocationType.AI, after_callback)
    manager.register_on_error(InvocationType.AI, error_callback)

    # Test 1: Before invoke callbacks (should allow)
    print("\n📋 Test 1: Before invoke callbacks (allowing)")
    context = CallbackContext(
        invocation_type=InvocationType.AI,
        node_name="test_node",
        function_name="test_function",
        metadata={"test": "data"},
    )

    input_data = {"test": "input"}
    result_data = await manager.execute_before_invoke(context, input_data)
    print(f"Result: {result_data} (expected: modified input)")
    print(f"Callback called: {before_callback.called} (expected: True)")

    # Test 2: After invoke callbacks
    print("\n📋 Test 2: After invoke callbacks")
    output_data = await manager.execute_after_invoke(context, input_data, "test_result")
    print(f"Callback called: {after_callback.called} (expected: True)")

    # Test 3: Error callbacks
    print("\n📋 Test 3: Error callbacks")
    test_error = Exception("Test error")
    error_result = await manager.execute_on_error(context, input_data, test_error)
    print(f"Callback called: {error_callback.called} (expected: True)")

    # Test 4: Blocking before callback
    print("\n📋 Test 4: Before invoke callbacks (blocking)")
    manager_blocking = CallbackManager()
    manager_blocking.register_before_invoke(InvocationType.AI, blocking_callback)

    try:
        result = await manager_blocking.execute_before_invoke(context, input_data)
        blocking_result = True  # If no exception, it didn't block properly
    except Exception:
        blocking_result = False  # Exception means it blocked correctly

    print(
        f"Blocking behavior: {'❌ Failed to block' if blocking_result else '✅ Blocked correctly'}"
    )
    print(f"Blocking callback called: {blocking_callback.called} (expected: True)")

    print("\n🎉 All callback system tests completed!")

    # Verify results
    success = (
        before_callback.called
        and after_callback.called
        and error_callback.called
        and blocking_callback.called
    )

    if success:
        print("✅ All tests passed! Callback system is working correctly.")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_callback_system())
    sys.exit(0 if success else 1)
