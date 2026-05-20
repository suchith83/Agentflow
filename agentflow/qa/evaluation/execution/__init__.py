"""
Execution result container for agent evaluation.

ExecutionResult holds tool calls, trajectory and response
built by AgentEvaluator._execution_from_collector() after ainvoke.
"""

from agentflow.qa.evaluation.token_usage import TokenUsage

from .result import ExecutionResult, NodeResponseData


__all__ = ["ExecutionResult", "NodeResponseData", "TokenUsage"]


__all__ = [
    "ExecutionResult",
    "NodeResponseData",
]
