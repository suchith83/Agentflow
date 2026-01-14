import asyncio

from agentflow.testing import QuickTest
from react_sync import agent

async def run_custom_test():
    """Run a custom test using the react_sync app."""
    print("\n🚀 Running Custom Test with react_sync App\n")

    result = await QuickTest.custom(
        agent=agent,
        user_message="hi there!", 
    )
    result.assert_contains("Hi there")
    result.assert_no_errors()

    print(f"✓ Custom test passed! Response: {result.final_response}\n")

async def main():
    await run_custom_test()

if __name__ == "__main__":
    asyncio.run(main())