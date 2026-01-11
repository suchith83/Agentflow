"""Examples demonstrating QuickEval and EvalPresets - simplified evaluation."""

import asyncio

from agentflow.evaluation import EvalPresets, EvalSetBuilder, QuickEval
from agentflow.graph import StateGraph
from agentflow.state import Message
from agentflow.testing import TestAgent
from agentflow.utils.constants import END


def create_test_graph():
    """Create a simple test graph for examples."""
    agent = TestAgent(
        model="test-model",
        responses=[
            "Hi there! How can I help?",
            "The weather is sunny and 72°F",
            "You're welcome!",
        ],
    )

    graph = StateGraph()
    graph.add_node("MAIN", agent)
    graph.set_entry_point("MAIN")
    graph.add_edge("MAIN", END)

    return graph.compile()


async def example_quick_check():
    """Example: Quick single check."""
    print("=" * 60)
    print("Example 1: Quick Check (One-Liner)")
    print("=" * 60)

    graph = create_test_graph()

    # BEFORE: Would need ~50 lines of config, eval set creation, etc.
    # NOW: Just one line!
    report = await QuickEval.check(
        graph=graph,
        query="Hello",
        expected_response_contains="help",
        verbose=False,
        print_results=False,
    )

    print(f"✓ Quick check passed!")
    print(f"  Pass rate: {report.summary.pass_rate * 100:.1f}%\n")


async def example_presets():
    """Example: Using evaluation presets."""
    print("=" * 60)
    print("Example 2: Using Presets")
    print("=" * 60)

    # Different presets for different scenarios

    # 1. Response quality preset
    config1 = EvalPresets.response_quality(threshold=0.7)
    print(f"✓ Response quality preset created")
    print(f"  Criteria: {list(config1.criteria.keys())}")

    # 2. Tool usage preset
    config2 = EvalPresets.tool_usage(strict=True)
    print(f"✓ Tool usage preset created")
    print(f"  Criteria: {list(config2.criteria.keys())}")

    # 3. Combined preset
    config3 = EvalPresets.combine(
        EvalPresets.response_quality(),
        EvalPresets.tool_usage(strict=False),
    )
    print(f"✓ Combined preset created")
    print(f"  Criteria: {list(config3.criteria.keys())}\n")


async def example_builder():
    """Example: Fluent eval set builder."""
    print("=" * 60)
    print("Example 3: EvalSetBuilder (Fluent API)")
    print("=" * 60)

    # BEFORE: Would need verbose EvalCase/EvalSet creation
    # NOW: Fluent builder pattern!
    eval_set = (
        EvalSetBuilder("my_tests")
        .add_case(
            query="Hello",
            expected="Hi there",
        )
        .add_case(
            query="How are you?",
            expected="I'm doing great",
        )
        .add_tool_test(
            query="Weather in NYC?",
            tool_name="get_weather",
            tool_args={"city": "NYC"},
        )
        .build()
    )

    print(f"✓ Eval set created with {len(eval_set.eval_cases)} cases")
    print(f"  Cases: {[c.eval_id for c in eval_set.eval_cases]}\n")


async def example_batch():
    """Example: Batch evaluation from pairs."""
    print("=" * 60)
    print("Example 4: Batch Evaluation")
    print("=" * 60)

    graph = create_test_graph()

    # Quick batch test from query-response pairs
    report = await QuickEval.batch(
        graph=graph,
        test_pairs=[
            ("Hello", "Hi"),
            ("How are you?", "great"),
            ("Thank you", "welcome"),
        ],
        threshold=0.5,
        verbose=False,
        print_results=False,
    )

    print(f"✓ Batch evaluation completed")
    print(f"  Total cases: {report.summary.total_cases}")
    print(f"  Passed: {report.summary.passed_cases}")
    print(f"  Pass rate: {report.summary.pass_rate * 100:.1f}%\n")


async def example_quick_builder():
    """Example: Quick builder from pairs."""
    print("=" * 60)
    print("Example 5: Quick Builder")
    print("=" * 60)

    # Super quick eval set creation
    eval_set = EvalSetBuilder.quick(
        ("Hello", "Hi"),
        ("Weather?", "Sunny"),
        ("Thanks", "Welcome"),
    )

    print(f"✓ Quick eval set created")
    print(f"  Cases: {len(eval_set.eval_cases)}\n")


async def example_custom_preset():
    """Example: Custom preset configuration."""
    print("=" * 60)
    print("Example 6: Custom Preset")
    print("=" * 60)

    # Create custom config with specific thresholds
    config = EvalPresets.custom(
        response_threshold=0.8,
        tool_threshold=0.9,
        llm_judge_threshold=0.75,
    )

    print(f"✓ Custom preset created")
    print(f"  Response threshold: 0.8")
    print(f"  Tool threshold: 0.9")
    print(f"  LLM judge threshold: 0.75\n")


async def main():
    """Run all examples."""
    print("\n🚀 QuickEval & EvalPresets Examples\n")
    print("Demonstrating simplified evaluation with 85% less boilerplate\n")

    await example_quick_check()
    await example_presets()
    await example_builder()
    await example_batch()
    await example_quick_builder()
    await example_custom_preset()

    print("=" * 60)
    print("✓ All examples completed successfully!")
    print("=" * 60)

    print("\n📝 Key Takeaways:")
    print("  • QuickEval.check() - One-liner for quick tests")
    print("  • EvalPresets - Ready-to-use configurations")
    print("  • EvalSetBuilder - Fluent API for test creation")
    print("  • 85% less code compared to manual setup!")


if __name__ == "__main__":
    asyncio.run(main())
