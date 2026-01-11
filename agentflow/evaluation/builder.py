"""Fluent builder for creating EvalSets easily."""

import uuid
from typing import Any

from agentflow.evaluation.eval_set import EvalCase, EvalSet, ToolCall


class EvalSetBuilder:
    """Fluent builder for creating evaluation sets.

    Provides a simple, chainable interface for building eval sets without
    verbose boilerplate.

    Example:
        ```python
        # Build eval set fluently
        eval_set = (
            EvalSetBuilder("my_tests")
            .add_case(query="Hello", expected="Hi there")
            .add_case(query="Weather?", expected="Sunny", expected_tools=["get_weather"])
            .build()
        )

        # Or from conversation logs
        eval_set = EvalSetBuilder.from_conversations(
            [
                {"user": "Hello", "assistant": "Hi!"},
                {"user": "How are you?", "assistant": "Great!"},
            ]
        )
        ```
    """

    def __init__(self, name: str = "eval_set"):
        """Initialize builder.

        Args:
            name: Name for the evaluation set
        """
        self.name = name
        self.eval_set_id = str(uuid.uuid4())
        self.cases: list[EvalCase] = []

    def add_case(
        self,
        query: str,
        expected: str,
        case_id: str | None = None,
        expected_tools: list[str | ToolCall] | None = None,
        **kwargs: Any,
    ) -> "EvalSetBuilder":
        """Add a test case to the eval set.

        Args:
            query: User query/input
            expected: Expected agent response
            case_id: Optional case ID (auto-generated if not provided)
            expected_tools: Expected tool calls (as names or ToolCall objects)
            **kwargs: Additional case parameters

        Returns:
            Self for method chaining
        """
        eval_id = case_id or f"case_{len(self.cases) + 1}"

        # Convert tool names to ToolCall objects
        tool_calls = []
        if expected_tools:
            for tool in expected_tools:
                if isinstance(tool, str):
                    tool_calls.append(ToolCall(name=tool, args={}))
                else:
                    tool_calls.append(tool)

        case = EvalCase.single_turn(
            eval_id=eval_id,
            user_query=query,
            expected_response=expected,
            expected_tools=tool_calls if tool_calls else None,
            **kwargs,
        )

        self.cases.append(case)
        return self

    def add_multi_turn(
        self,
        conversation: list[tuple[str, str]],
        case_id: str | None = None,
        expected_tools: list[str | ToolCall] | None = None,
        **kwargs: Any,
    ) -> "EvalSetBuilder":
        """Add a multi-turn conversation case.

        Args:
            conversation: List of (user_query, expected_response) tuples
            case_id: Optional case ID
            expected_tools: Expected tool calls across conversation
            **kwargs: Additional case parameters

        Returns:
            Self for method chaining
        """
        eval_id = case_id or f"case_{len(self.cases) + 1}"

        # Convert tool names to ToolCall objects
        tool_calls = []
        if expected_tools:
            for tool in expected_tools:
                if isinstance(tool, str):
                    tool_calls.append(ToolCall(name=tool, args={}))
                else:
                    tool_calls.append(tool)

        case = EvalCase.multi_turn(
            eval_id=eval_id,
            conversation=conversation,
            expected_tools=tool_calls if tool_calls else None,
            **kwargs,
        )

        self.cases.append(case)
        return self

    def add_tool_test(
        self,
        query: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        expected_response: str | None = None,
        case_id: str | None = None,
    ) -> "EvalSetBuilder":
        """Add a test case focused on tool usage.

        Args:
            query: User query
            tool_name: Expected tool to be called
            tool_args: Expected tool arguments
            expected_response: Expected final response (optional)
            case_id: Optional case ID

        Returns:
            Self for method chaining
        """
        eval_id = case_id or f"case_{len(self.cases) + 1}"

        tool_call = ToolCall(name=tool_name, args=tool_args or {})

        case = EvalCase.single_turn(
            eval_id=eval_id,
            user_query=query,
            expected_response=expected_response or f"Result from {tool_name}",
            expected_tools=[tool_call],
        )

        self.cases.append(case)
        return self

    def build(self) -> EvalSet:
        """Build and return the final EvalSet.

        Returns:
            Complete EvalSet with all added cases
        """
        return EvalSet(
            eval_set_id=self.eval_set_id,
            name=self.name,
            eval_cases=self.cases,
        )

    def save(self, path: str) -> EvalSet:
        """Build and save the EvalSet to a file.

        Args:
            path: File path to save JSON

        Returns:
            The built EvalSet
        """
        eval_set = self.build()
        eval_set.save(path)
        return eval_set

    @classmethod
    def from_conversations(
        cls,
        conversations: list[dict[str, str]],
        name: str = "conversation_tests",
    ) -> EvalSet:
        """Create eval set from conversation logs.

        Args:
            conversations: List of conversation dicts with 'user' and 'assistant' keys
            name: Name for the eval set

        Returns:
            Built EvalSet
        """
        builder = cls(name)

        for i, conv in enumerate(conversations):
            user_msg = conv.get("user", "")
            assistant_msg = conv.get("assistant", "")

            builder.add_case(
                query=user_msg,
                expected=assistant_msg,
                case_id=f"conv_{i + 1}",
            )

        return builder.build()

    @classmethod
    def from_file(cls, path: str) -> "EvalSetBuilder":
        """Load existing eval set and convert to builder for modification.

        Args:
            path: Path to existing eval set JSON

        Returns:
            Builder initialized with cases from file
        """
        eval_set = EvalSet.from_file(path)

        builder = cls(eval_set.name or "eval_set")
        builder.eval_set_id = eval_set.eval_set_id
        builder.cases = eval_set.eval_cases

        return builder

    @classmethod
    def quick(cls, *test_pairs: tuple[str, str]) -> EvalSet:
        """Quick builder from test pairs.

        Args:
            *test_pairs: Variable number of (query, expected) tuples

        Returns:
            Built EvalSet

        Example:
            ```python
            eval_set = EvalSetBuilder.quick(
                ("Hello", "Hi!"),
                ("How are you?", "Great!"),
            )
            ```
        """
        builder = cls("quick_tests")

        for i, (query, expected) in enumerate(test_pairs):
            builder.add_case(
                query=query,
                expected=expected,
                case_id=f"test_{i + 1}",
            )

        return builder.build()
