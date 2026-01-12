"""
Sample evaluation configuration for the react_sync agent.

This module demonstrates how to set up comprehensive evaluation cases
for the ReAct agent using the agentflow evaluation framework.
"""

from typing import List

from agentflow.evaluation import (
    CriterionConfig,
    EvalCase,
    EvalConfig,
    MatchType,
    MessageContent,
    ToolCall,
    TrajectoryStep,
)


def create_weather_evaluation_cases() -> List[EvalCase]:
    """
    Create evaluation cases for weather-related queries.

    Returns:
        List of EvalCase objects for testing weather tool usage.
    """
    cases = []

    # Case 1: Simple weather query
    cases.append(
        EvalCase(
            id="weather_simple_001",
            name="Simple weather query for NYC",
            description="Agent should call weather tool for New York City",
            input_messages=[
                MessageContent(role="user", content="What's the weather in New York City?")
            ],
            expected_trajectory=[
                TrajectoryStep(
                    node_name="MAIN",
                    tool_calls=[
                        ToolCall(name="get_weather", arguments={"location": "New York City"})
                    ],
                ),
                TrajectoryStep(node_name="TOOL", tool_calls=[]),
                TrajectoryStep(node_name="MAIN", tool_calls=[]),
            ],
            expected_tool_calls=["get_weather"],
            match_type=MatchType.IN_ORDER,
            metadata={"category": "weather", "difficulty": "easy", "expected_tool_count": 1},
        )
    )

    # Case 2: Explicit tool call request
    cases.append(
        EvalCase(
            id="weather_explicit_002",
            name="Explicit request to call weather function",
            description="User explicitly asks to call get_weather function",
            input_messages=[
                MessageContent(
                    role="user", content="Please call the get_weather function for Tokyo"
                )
            ],
            expected_trajectory=[
                TrajectoryStep(
                    node_name="MAIN",
                    tool_calls=[ToolCall(name="get_weather", arguments={"location": "Tokyo"})],
                ),
                TrajectoryStep(node_name="TOOL"),
                TrajectoryStep(node_name="MAIN"),
            ],
            expected_tool_calls=["get_weather"],
            match_type=MatchType.EXACT,
            metadata={"category": "weather", "difficulty": "easy", "expected_tool_count": 1},
        )
    )

    # Case 3: Multiple cities (should call tool multiple times or handle differently)
    cases.append(
        EvalCase(
            id="weather_multiple_003",
            name="Weather for multiple cities",
            description="Agent should handle requests for multiple cities",
            input_messages=[
                MessageContent(
                    role="user", content="Can you tell me the weather in London and Paris?"
                )
            ],
            expected_trajectory=[
                TrajectoryStep(
                    node_name="MAIN",
                    tool_calls=[
                        ToolCall(name="get_weather", arguments={"location": "London"}),
                    ],
                ),
                TrajectoryStep(node_name="TOOL"),
                TrajectoryStep(
                    node_name="MAIN",
                    tool_calls=[
                        ToolCall(name="get_weather", arguments={"location": "Paris"}),
                    ],
                ),
                TrajectoryStep(node_name="TOOL"),
                TrajectoryStep(node_name="MAIN"),
            ],
            expected_tool_calls=["get_weather", "get_weather"],
            match_type=MatchType.UNORDERED,
            metadata={"category": "weather", "difficulty": "medium", "expected_tool_count": 2},
        )
    )

    # Case 4: Conversational weather query
    cases.append(
        EvalCase(
            id="weather_conversational_004",
            name="Conversational weather inquiry",
            description="More natural conversation about weather",
            input_messages=[
                MessageContent(
                    role="user",
                    content="I'm planning to visit Berlin tomorrow. Should I bring an umbrella?",
                )
            ],
            expected_trajectory=[
                TrajectoryStep(
                    node_name="MAIN",
                    tool_calls=[ToolCall(name="get_weather", arguments={"location": "Berlin"})],
                ),
                TrajectoryStep(node_name="TOOL"),
                TrajectoryStep(node_name="MAIN"),
            ],
            expected_tool_calls=["get_weather"],
            match_type=MatchType.IN_ORDER,
            metadata={"category": "weather", "difficulty": "medium", "requires_reasoning": True},
        )
    )

    # Case 5: Invalid location handling
    cases.append(
        EvalCase(
            id="weather_edge_005",
            name="Weather for unusual location",
            description="Agent should handle unusual location names gracefully",
            input_messages=[MessageContent(role="user", content="What's the weather in Narnia?")],
            expected_trajectory=[
                TrajectoryStep(
                    node_name="MAIN",
                    tool_calls=[ToolCall(name="get_weather", arguments={"location": "Narnia"})],
                ),
                TrajectoryStep(node_name="TOOL"),
                TrajectoryStep(node_name="MAIN"),
            ],
            expected_tool_calls=["get_weather"],
            match_type=MatchType.IN_ORDER,
            metadata={"category": "edge_case", "difficulty": "easy", "expected_tool_count": 1},
        )
    )

    return cases


def create_routing_evaluation_cases() -> List[EvalCase]:
    """
    Create evaluation cases for testing routing logic.

    Returns:
        List of EvalCase objects for testing graph routing.
    """
    cases = []

    # Case 1: No tool needed
    cases.append(
        EvalCase(
            id="routing_no_tool_001",
            name="Query that doesn't need tools",
            description="Agent should respond without calling tools",
            input_messages=[MessageContent(role="user", content="What is your purpose?")],
            expected_trajectory=[TrajectoryStep(node_name="MAIN", tool_calls=[])],
            expected_tool_calls=[],
            match_type=MatchType.EXACT,
            metadata={"category": "routing", "difficulty": "easy", "expected_tool_count": 0},
        )
    )

    # Case 2: Direct tool request
    cases.append(
        EvalCase(
            id="routing_direct_tool_002",
            name="Direct tool invocation",
            description="User directly requests tool usage",
            input_messages=[MessageContent(role="user", content="Call get_weather for Sydney")],
            expected_trajectory=[
                TrajectoryStep(
                    node_name="MAIN",
                    tool_calls=[ToolCall(name="get_weather", arguments={"location": "Sydney"})],
                ),
                TrajectoryStep(node_name="TOOL"),
                TrajectoryStep(node_name="MAIN"),
            ],
            expected_tool_calls=["get_weather"],
            match_type=MatchType.IN_ORDER,
            metadata={"category": "routing", "difficulty": "easy", "expected_tool_count": 1},
        )
    )

    return cases


def create_comprehensive_eval_config() -> EvalConfig:
    """
    Create a comprehensive evaluation configuration.

    Returns:
        EvalConfig object with all evaluation cases and criteria.
    """
    # Combine all evaluation cases
    all_cases = []
    all_cases.extend(create_weather_evaluation_cases())
    all_cases.extend(create_routing_evaluation_cases())

    # Create criterion configurations
    trajectory_criterion = CriterionConfig(
        name="trajectory_match", threshold=0.8, match_type=MatchType.IN_ORDER, weight=1.0
    )

    tool_call_criterion = CriterionConfig(
        name="tool_call_accuracy", threshold=1.0, match_type=MatchType.EXACT, weight=1.5
    )

    response_quality_criterion = CriterionConfig(name="response_quality", threshold=0.7, weight=0.8)

    # Create evaluation configuration
    config = EvalConfig(
        eval_set_id="react_sync_comprehensive",
        eval_set_name="React Sync Agent - Comprehensive Evaluation",
        eval_cases=all_cases,
        criteria_configs=[trajectory_criterion, tool_call_criterion, response_quality_criterion],
        metadata={
            "version": "1.0",
            "model": "gemini-2.5-flash",
            "provider": "google",
            "description": "Comprehensive evaluation suite for react_sync.py example",
        },
    )

    return config


# Example usage functions


def get_all_evaluation_cases() -> List[EvalCase]:
    """Get all evaluation cases for the react_sync agent."""
    cases = []
    cases.extend(create_weather_evaluation_cases())
    cases.extend(create_routing_evaluation_cases())
    return cases


def get_weather_cases_only() -> List[EvalCase]:
    """Get only weather-related evaluation cases."""
    return create_weather_evaluation_cases()


def get_routing_cases_only() -> List[EvalCase]:
    """Get only routing-related evaluation cases."""
    return create_routing_evaluation_cases()


def filter_cases_by_difficulty(cases: List[EvalCase], difficulty: str) -> List[EvalCase]:
    """
    Filter evaluation cases by difficulty level.

    Args:
        cases: List of evaluation cases
        difficulty: Difficulty level ('easy', 'medium', 'hard')

    Returns:
        Filtered list of evaluation cases
    """
    return [case for case in cases if case.metadata.get("difficulty") == difficulty]


def filter_cases_by_category(cases: List[EvalCase], category: str) -> List[EvalCase]:
    """
    Filter evaluation cases by category.

    Args:
        cases: List of evaluation cases
        category: Category name ('weather', 'routing', 'edge_case', etc.)

    Returns:
        Filtered list of evaluation cases
    """
    return [case for case in cases if case.metadata.get("category") == category]


if __name__ == "__main__":
    # Example: Print all evaluation cases
    config = create_comprehensive_eval_config()

    print(f"Evaluation Set: {config.eval_set_name}")
    print(f"Total Cases: {len(config.eval_cases)}")
    print("\n" + "=" * 70 + "\n")

    for i, case in enumerate(config.eval_cases, 1):
        print(f"{i}. {case.name}")
        print(f"   ID: {case.id}")
        print(f"   Category: {case.metadata.get('category', 'N/A')}")
        print(f"   Difficulty: {case.metadata.get('difficulty', 'N/A')}")
        print(
            f"   Expected Tools: {', '.join(case.expected_tool_calls) if case.expected_tool_calls else 'None'}"
        )
        print()

    print("=" * 70)
    print(f"\nCriteria Configurations: {len(config.criteria_configs)}")
    for criterion in config.criteria_configs:
        print(
            f"  - {criterion.name} (threshold: {criterion.threshold}, weight: {criterion.weight})"
        )
