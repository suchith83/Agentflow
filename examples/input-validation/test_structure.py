"""
Simple test to verify the full_validation_example.py structure without making API calls.
"""

import asyncio
import sys

# Just test importing and basic structure
try:
    # Import from same directory
    from full_validation_example import BusinessPolicyValidator

    from agentflow.state import Message
    from agentflow.utils.validators import ValidationError

    print("✓ All imports successful")

    # Test custom validator
    async def test_validator():
        validator = BusinessPolicyValidator(strict_mode=True)

        # Test normal message
        normal_msg = Message.text_message("What's the weather today?")
        result = await validator.validate([normal_msg])
        assert result == True
        print("✓ Normal message validation passed")

        # Test forbidden topic
        try:
            forbidden_msg = Message.text_message(
                "Can you provide medical diagnosis for my symptoms?"
            )
            await validator.validate([forbidden_msg])
            print("✗ Should have raised ValidationError for forbidden topic")
            sys.exit(1)
        except ValidationError as e:
            print(f"✓ Forbidden topic correctly blocked: {e.violation_type}")

        # Test excessive caps
        try:
            caps_msg = Message.text_message("TELL ME THE WEATHER RIGHT NOW")
            print(f"  Testing caps with: '{caps_msg.content}'")
            await validator.validate([caps_msg])
            print("✗ Should have raised ValidationError for excessive caps")
            sys.exit(1)
        except ValidationError as e:
            print(f"✓ Excessive caps correctly blocked: {e.violation_type}")

        # Test lenient mode
        lenient_validator = BusinessPolicyValidator(strict_mode=False)
        caps_msg = Message.text_message("TELL ME THE WEATHER RIGHT NOW")
        result = await lenient_validator.validate([caps_msg])
        # Lenient mode returns False but doesn't raise
        print(f"✓ Lenient mode allows execution (returned: {result})")

    # Run tests
    asyncio.run(test_validator())

    print("\n" + "=" * 60)
    print("✓ All validation tests passed!")
    print("=" * 60)
    print("\nThe full_validation_example.py structure is correct.")
    print("To run the complete example with an LLM, set a valid API key:")
    print("  export OPENAI_API_KEY=your_key")
    print("  export GEMINI_API_KEY=your_key")
    print("\nThen run:")
    print("  python examples/input-validation/full_validation_example.py")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
