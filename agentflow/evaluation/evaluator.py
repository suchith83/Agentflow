"""
Main AgentEvaluator class for running agent evaluations.

This module provides the central entry point for evaluating agent graphs
against test cases defined in EvalSet files.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentflow.evaluation.collectors.trajectory_collector import TrajectoryCollector
from agentflow.evaluation.criteria.base import BaseCriterion
from agentflow.evaluation.criteria.llm_judge import LLMJudgeCriterion, RubricBasedCriterion
from agentflow.evaluation.criteria.response import ResponseMatchCriterion
from agentflow.evaluation.criteria.trajectory import TrajectoryMatchCriterion
from agentflow.evaluation.eval_config import CriterionConfig, EvalConfig
from agentflow.evaluation.eval_result import (
    CriterionResult,
    EvalCaseResult,
    EvalReport,
)
from agentflow.evaluation.eval_set import EvalCase, EvalSet


if TYPE_CHECKING:
    from agentflow.graph.compiled_graph import CompiledGraph

logger = logging.getLogger("agentflow.evaluation")


class AgentEvaluator:
    """Main class for running agent evaluations.

    The AgentEvaluator orchestrates the evaluation process by:
    1. Loading evaluation sets from files or accepting EvalSet objects
    2. Running the agent graph for each test case
    3. Collecting execution trajectories and responses
    4. Evaluating against configured criteria
    5. Generating comprehensive reports

    Attributes:
        graph: The compiled agent graph to evaluate.
        config: Evaluation configuration with criteria settings.
        criteria: List of active criteria for evaluation.

    Example:
        ```python
        from agentflow.evaluation import AgentEvaluator, EvalConfig

        # Create evaluator
        evaluator = AgentEvaluator(compiled_graph, config=EvalConfig.default())

        # Run evaluation from file
        report = await evaluator.evaluate("tests/fixtures/my_tests.evalset.json")

        # Or from EvalSet object
        report = await evaluator.evaluate(eval_set)

        # Print summary
        print(report.format_summary())
        ```
    """

    def __init__(
        self,
        graph: CompiledGraph,
        config: EvalConfig | None = None,
    ):
        """Initialize the evaluator.

        Args:
            graph: The compiled agent graph to evaluate.
            config: Optional evaluation configuration. Uses defaults if not provided.
        """
        self.graph = graph
        self.config = config or EvalConfig.default()
        self.criteria = self._build_criteria()

    def _build_criteria(self) -> list[BaseCriterion]:
        """Build criteria instances from configuration."""
        criteria = []

        for name, criterion_config in self.config.criteria.items():
            if not criterion_config.enabled:
                continue

            criterion = self._create_criterion(name, criterion_config)
            if criterion:
                criteria.append(criterion)

        return criteria

    def _create_criterion(
        self,
        name: str,
        criterion_config: CriterionConfig,
    ) -> BaseCriterion | None:
        """Create a criterion instance by name.

        Args:
            name: The criterion name/type.
            criterion_config: Configuration for the criterion.

        Returns:
            A configured criterion instance, or None if unknown.
        """
        criterion_map = {
            "tool_trajectory_avg_score": TrajectoryMatchCriterion,
            "trajectory_match": TrajectoryMatchCriterion,
            "response_match_score": ResponseMatchCriterion,
            "response_match": ResponseMatchCriterion,
            "final_response_match_v2": LLMJudgeCriterion,
            "llm_judge": LLMJudgeCriterion,
            "rubric_based_final_response_quality_v1": RubricBasedCriterion,
            "rubric_based": RubricBasedCriterion,
        }

        criterion_class = criterion_map.get(name)
        if criterion_class:
            return criterion_class(config=criterion_config)

        logger.warning("Unknown criterion: %s", name)
        return None

    async def evaluate(
        self,
        eval_set: EvalSet | str,
        parallel: bool = False,
        max_concurrency: int = 4,
        verbose: bool = False,
    ) -> EvalReport:
        """Run evaluation on an eval set.

        Args:
            eval_set: EvalSet object or path to JSON file.
            parallel: Whether to run cases in parallel.
            max_concurrency: Maximum concurrent case evaluations.
            verbose: Whether to log detailed progress.

        Returns:
            Complete evaluation report with results and summary.
        """
        # Load eval set if path provided
        if isinstance(eval_set, str):
            eval_set = self._load_eval_set(eval_set)

        if verbose:
            logger.info(
                "Starting evaluation of '%s' with %d cases",
                eval_set.name or eval_set.eval_set_id,
                len(eval_set.eval_cases),
            )

        # Run evaluation
        if parallel and len(eval_set.eval_cases) > 1:
            results = await self._evaluate_parallel(
                eval_set.eval_cases,
                max_concurrency,
                verbose,
            )
        else:
            results = await self._evaluate_sequential(
                eval_set.eval_cases,
                verbose,
            )

        # Create report
        report = EvalReport.create(
            eval_set_id=eval_set.eval_set_id,
            results=results,
            eval_set_name=eval_set.name,
            config_used=self.config.model_dump(),
        )

        if verbose:
            logger.info(
                "Evaluation complete: %d/%d passed (%.1f%%)",
                report.summary.passed_cases,
                report.summary.total_cases,
                report.summary.pass_rate * 100,
            )

        return report

    async def _evaluate_sequential(
        self,
        cases: list[EvalCase],
        verbose: bool = False,
    ) -> list[EvalCaseResult]:
        """Evaluate cases sequentially."""
        results = []

        for i, case in enumerate(cases):
            if verbose:
                logger.info(
                    "Evaluating case %d/%d: %s",
                    i + 1,
                    len(cases),
                    case.name or case.eval_id,
                )

            result = await self._evaluate_case(case)
            results.append(result)

            if verbose:
                status = "PASSED" if result.passed else "FAILED"
                logger.info("  -> %s (%.2fs)", status, result.duration_seconds)

        return results

    async def _evaluate_parallel(
        self,
        cases: list[EvalCase],
        max_concurrency: int,
        verbose: bool = False,
    ) -> list[EvalCaseResult]:
        """Evaluate cases in parallel with concurrency limit."""
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrency)

        async def evaluate_with_semaphore(case: EvalCase) -> EvalCaseResult:
            async with semaphore:
                return await self._evaluate_case(case)

        tasks = [evaluate_with_semaphore(case) for case in cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failure results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    EvalCaseResult.failure(
                        eval_id=cases[i].eval_id,
                        error=str(result),
                        name=cases[i].name,
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _evaluate_case(self, case: EvalCase) -> EvalCaseResult:
        """Evaluate a single test case.

        Args:
            case: The evaluation case to run.

        Returns:
            Result of the case evaluation.
        """
        start_time = time.time()

        try:
            # Create collector for this execution
            collector = TrajectoryCollector(capture_all_events=True)

            # Prepare input state
            from agentflow.state import Message

            state_dict: dict[str, Any] = {}

            # Build messages from conversation
            for invocation in case.conversation:
                # Create user message
                user_text = invocation.user_content.get_text()
                user_msg = Message.text_message(user_text, role="user")

                if "messages" not in state_dict:
                    state_dict["messages"] = []
                state_dict["messages"].append(user_msg)

            # Execute graph with trajectory collection
            config = {
                "callbacks": [collector.on_event],
                "user_id": case.session_input.user_id,
                **case.session_input.config,
            }

            try:
                result = await self.graph.ainvoke(state_dict, config=config)
            except Exception as e:
                logger.warning("Graph execution failed for case %s: %s", case.eval_id, e)
                duration = time.time() - start_time
                return EvalCaseResult.failure(
                    eval_id=case.eval_id,
                    error=f"Graph execution error: {e}",
                    name=case.name,
                    duration_seconds=duration,
                )

            # Extract actual response from result
            actual_response = self._extract_response(result)
            collector.messages.append(
                {
                    "role": "assistant",
                    "content": actual_response,
                }
            )

            # Evaluate against all criteria
            criterion_results = []
            for criterion in self.criteria:
                try:
                    cr_result = await criterion.evaluate(collector, case)
                    criterion_results.append(cr_result)
                except Exception as e:
                    logger.error(
                        "Criterion %s failed for case %s: %s",
                        criterion.name,
                        case.eval_id,
                        e,
                    )
                    criterion_results.append(
                        CriterionResult.failure(
                            criterion=criterion.name,
                            error=str(e),
                        )
                    )

            duration = time.time() - start_time

            return EvalCaseResult.success(
                eval_id=case.eval_id,
                criterion_results=criterion_results,
                actual_trajectory=collector.trajectory,
                actual_tool_calls=collector.tool_calls,
                actual_response=actual_response,
                duration_seconds=duration,
                name=case.name,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error("Case evaluation failed: %s", e)
            return EvalCaseResult.failure(
                eval_id=case.eval_id,
                error=str(e),
                name=case.name,
                duration_seconds=duration,
            )

    def _extract_response(self, result: dict[str, Any]) -> str:
        """Extract the final response text from graph result."""
        if not result:
            return ""

        # Check for messages in result
        messages = result.get("messages", [])
        if messages:
            # Get last assistant message
            for msg in reversed(messages):
                if hasattr(msg, "role") and msg.role == "assistant":
                    return msg.get_text() if hasattr(msg, "get_text") else str(msg)
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        texts = [
                            b.get("text", "")
                            for b in content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        return " ".join(texts)

        # Check for direct content
        if "content" in result:
            return str(result["content"])

        return ""

    def _load_eval_set(self, path: str) -> EvalSet:
        """Load an EvalSet from a file.

        Args:
            path: Path to the JSON file.

        Returns:
            Loaded EvalSet.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is invalid.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Eval set file not found: {path}")

        return EvalSet.from_file(str(file_path))

    @classmethod
    def evaluate_sync(
        cls,
        graph: CompiledGraph,
        eval_set: EvalSet | str,
        config: EvalConfig | None = None,
        verbose: bool = False,
    ) -> EvalReport:
        """Synchronous wrapper for evaluate().

        Args:
            graph: The compiled graph to evaluate.
            eval_set: EvalSet object or path to JSON file.
            config: Optional evaluation configuration.
            verbose: Whether to log progress.

        Returns:
            Complete evaluation report.
        """
        evaluator = cls(graph, config)
        return asyncio.run(evaluator.evaluate(eval_set, verbose=verbose))

    @classmethod
    async def evaluate_file(
        cls,
        agent_module: str,
        eval_file: str,
        config_file: str | None = None,
    ) -> EvalReport:
        """Convenience method to evaluate from file paths.

        This method loads the agent graph from a module and runs evaluation
        against the specified eval file.

        Args:
            agent_module: Python module path containing 'graph' variable.
            eval_file: Path to the eval set JSON file.
            config_file: Optional path to eval config JSON file.

        Returns:
            Complete evaluation report.

        Example:
            ```python
            report = await AgentEvaluator.evaluate_file(
                agent_module="examples.react.react_weather_agent",
                eval_file="tests/fixtures/weather_agent.evalset.json",
                config_file="tests/fixtures/eval_config.json",
            )
            ```
        """
        graph = cls._load_graph(agent_module)
        config = cls._load_config(config_file) if config_file else None
        evaluator = cls(graph, config)
        return await evaluator.evaluate(eval_file)

    @classmethod
    def _load_graph(cls, agent_module: str) -> CompiledGraph:
        """Load a compiled graph from a module.

        Args:
            agent_module: Python module path (e.g., "examples.react.agent").

        Returns:
            The compiled graph from the module.

        Raises:
            ImportError: If module cannot be imported.
            AttributeError: If module has no 'graph' attribute.
        """
        import importlib

        module = importlib.import_module(agent_module)

        # Look for common graph variable names
        for attr_name in ["graph", "compiled_graph", "agent_graph"]:
            if hasattr(module, attr_name):
                graph = getattr(module, attr_name)
                # If it's a StateGraph, compile it
                if hasattr(graph, "compile"):
                    return graph.compile()
                return graph

        raise AttributeError(f"Module {agent_module} has no 'graph' attribute")

    @classmethod
    def _load_config(cls, config_file: str) -> EvalConfig:
        """Load evaluation config from a file.

        Args:
            config_file: Path to the JSON config file.

        Returns:
            Loaded EvalConfig.
        """
        import json
        from pathlib import Path

        data = json.loads(Path(config_file).read_text(encoding="utf-8"))
        return EvalConfig.model_validate(data)


class EvaluationRunner:
    """Batch evaluation runner for multiple eval sets.

    Useful for running comprehensive evaluation suites across
    multiple agents and test sets.
    """

    def __init__(self, default_config: EvalConfig | None = None):
        """Initialize the runner.

        Args:
            default_config: Default configuration for evaluations.
        """
        self.default_config = default_config or EvalConfig.default()
        self.results: dict[str, EvalReport] = {}

    async def run(
        self,
        evaluations: list[tuple[CompiledGraph, EvalSet | str]],
        config: EvalConfig | None = None,
        verbose: bool = False,
    ) -> dict[str, EvalReport]:
        """Run multiple evaluations.

        Args:
            evaluations: List of (graph, eval_set) tuples.
            config: Override configuration for all evaluations.
            verbose: Whether to log progress.

        Returns:
            Dictionary mapping eval_set_id to report.
        """
        cfg = config or self.default_config

        for graph, eval_set in evaluations:
            evaluator = AgentEvaluator(graph, cfg)
            report = await evaluator.evaluate(eval_set, verbose=verbose)

            self.results[report.eval_set_id] = report

        return self.results

    @property
    def summary(self) -> dict[str, Any]:
        """Get aggregate summary across all evaluations."""
        if not self.results:
            return {"total_evaluations": 0}

        total_cases = sum(r.summary.total_cases for r in self.results.values())
        passed_cases = sum(r.summary.passed_cases for r in self.results.values())

        return {
            "total_evaluations": len(self.results),
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "overall_pass_rate": passed_cases / total_cases if total_cases > 0 else 0.0,
        }
