"""
Trajectory matching criterion.

Compares actual tool call trajectories against expected trajectories
using different matching strategies (EXACT, IN_ORDER, ANY_ORDER).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentflow.evaluation.criteria.base import SyncCriterion
from agentflow.evaluation.eval_config import MatchType
from agentflow.evaluation.eval_result import CriterionResult
from agentflow.evaluation.eval_set import ToolCall

if TYPE_CHECKING:
    from agentflow.evaluation.collectors.trajectory_collector import TrajectoryCollector
    from agentflow.evaluation.eval_config import CriterionConfig
    from agentflow.evaluation.eval_set import EvalCase


class TrajectoryMatchCriterion(SyncCriterion):
    """Compare actual vs expected tool call trajectories.

    This criterion evaluates whether the agent called the expected tools
    in the expected order (or any order, depending on configuration).

    Supports three matching modes:
        - EXACT: Require perfect match - same tools, same order, no extras
        - IN_ORDER: Expected tools must appear in order, extras allowed
        - ANY_ORDER: Expected tools must appear, any order, extras allowed

    Attributes:
        name: "tool_trajectory_avg_score"
        match_type: The matching strategy to use
        check_args: Whether to also compare tool arguments

    Example:
        ```python
        criterion = TrajectoryMatchCriterion(
            config=CriterionConfig(
                threshold=1.0,
                match_type=MatchType.IN_ORDER,
                check_args=False,
            )
        )

        result = await criterion.evaluate(collector, eval_case)
        print(f"Trajectory score: {result.score}")
        ```
    """

    name = "tool_trajectory_avg_score"
    description = "Evaluates tool call trajectory matching"

    def evaluate_sync(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
    ) -> CriterionResult:
        """Evaluate trajectory match between actual and expected.

        Args:
            actual: Collected trajectory from actual execution.
            expected: The evaluation case with expected outcomes.

        Returns:
            CriterionResult with trajectory match score.
        """
        # Collect expected tools from all invocations
        expected_tools: list[ToolCall] = []
        for invocation in expected.conversation:
            expected_tools.extend(invocation.expected_tool_trajectory)

        # Get actual tool calls
        actual_tools = actual.tool_calls

        # Calculate score based on match type
        match_type = self.config.match_type
        check_args = self.config.check_args

        if match_type == MatchType.EXACT:
            score = self._exact_match(actual_tools, expected_tools, check_args)
        elif match_type == MatchType.IN_ORDER:
            score = self._in_order_match(actual_tools, expected_tools, check_args)
        else:  # ANY_ORDER
            score = self._any_order_match(actual_tools, expected_tools, check_args)

        return CriterionResult.success(
            criterion=self.name,
            score=score,
            threshold=self.threshold,
            details={
                "actual_trajectory": [t.model_dump() for t in actual_tools],
                "expected_trajectory": [t.model_dump() for t in expected_tools],
                "match_type": match_type.value,
                "check_args": check_args,
            },
        )

    def _tools_match(
        self,
        actual: ToolCall,
        expected: ToolCall,
        check_args: bool,
    ) -> bool:
        """Check if two tool calls match."""
        return actual.matches(expected, check_args=check_args, check_call_id=False)

    def _exact_match(
        self,
        actual: list[ToolCall],
        expected: list[ToolCall],
        check_args: bool,
    ) -> float:
        """Require perfect match - same tools, same order, same length.

        Args:
            actual: List of actual tool calls.
            expected: List of expected tool calls.
            check_args: Whether to compare arguments.

        Returns:
            Score from 0.0 to 1.0.
        """
        if not expected:
            # If no tools expected, perfect match if none called
            return 1.0 if not actual else 0.0

        if len(actual) != len(expected):
            # Different lengths means partial match at best
            min_len = min(len(actual), len(expected))
            matches = sum(
                1
                for a, e in zip(actual[:min_len], expected[:min_len])
                if self._tools_match(a, e, check_args)
            )
            return matches / len(expected)

        # Same length - count matches
        matches = sum(1 for a, e in zip(actual, expected) if self._tools_match(a, e, check_args))
        return matches / len(expected)

    def _in_order_match(
        self,
        actual: list[ToolCall],
        expected: list[ToolCall],
        check_args: bool,
    ) -> float:
        """Check if expected tools appear in order, extras allowed.

        Uses a greedy approach to find expected tools in order within
        the actual trajectory.

        Args:
            actual: List of actual tool calls.
            expected: List of expected tool calls.
            check_args: Whether to compare arguments.

        Returns:
            Score from 0.0 to 1.0 (fraction of expected tools found in order).
        """
        if not expected:
            return 1.0

        expected_idx = 0
        for actual_tool in actual:
            if expected_idx >= len(expected):
                break
            if self._tools_match(actual_tool, expected[expected_idx], check_args):
                expected_idx += 1

        return expected_idx / len(expected)

    def _any_order_match(
        self,
        actual: list[ToolCall],
        expected: list[ToolCall],
        check_args: bool,
    ) -> float:
        """Check if expected tools appear in any order, extras allowed.

        Args:
            actual: List of actual tool calls.
            expected: List of expected tool calls.
            check_args: Whether to compare arguments.

        Returns:
            Score from 0.0 to 1.0 (fraction of expected tools found).
        """
        if not expected:
            return 1.0

        matched = 0
        remaining_actual = list(actual)

        for exp_tool in expected:
            for idx, act_tool in enumerate(remaining_actual):
                if self._tools_match(act_tool, exp_tool, check_args):
                    matched += 1
                    remaining_actual.pop(idx)
                    break

        return matched / len(expected)


class ToolNameMatchCriterion(SyncCriterion):
    """Simpler criterion that only checks tool names, not arguments.

    Useful for quick sanity checks where argument matching is too strict.
    """

    name = "tool_name_match_score"
    description = "Evaluates tool name matching (ignores arguments)"

    def evaluate_sync(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
    ) -> CriterionResult:
        """Evaluate tool name match."""
        # Collect expected tool names
        expected_names: list[str] = []
        for invocation in expected.conversation:
            expected_names.extend(t.name for t in invocation.expected_tool_trajectory)

        # Get actual tool names
        actual_names = actual.get_tool_names()

        if not expected_names:
            score = 1.0 if not actual_names else 0.5  # Penalize unexpected tools slightly
        else:
            # Count how many expected tools were called
            matched = 0
            remaining = list(actual_names)
            for exp_name in expected_names:
                if exp_name in remaining:
                    matched += 1
                    remaining.remove(exp_name)
            score = matched / len(expected_names)

        return CriterionResult.success(
            criterion=self.name,
            score=score,
            threshold=self.threshold,
            details={
                "expected_names": expected_names,
                "actual_names": actual_names,
            },
        )
