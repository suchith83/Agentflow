"""
LLM-as-Judge criterion.

Uses an LLM to evaluate semantic similarity and quality of responses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentflow.evaluation.criteria.base import BaseCriterion
from agentflow.evaluation.eval_result import CriterionResult

if TYPE_CHECKING:
    from agentflow.evaluation.collectors.trajectory_collector import TrajectoryCollector
    from agentflow.evaluation.eval_config import CriterionConfig
    from agentflow.evaluation.eval_set import EvalCase

logger = logging.getLogger("agentflow.evaluation")


SEMANTIC_MATCH_PROMPT = """You are an expert evaluator. Your task is to determine if the ACTUAL response is semantically equivalent to the EXPECTED response.

QUESTION: {question}

EXPECTED RESPONSE: {expected}

ACTUAL RESPONSE: {actual}

Evaluate if the ACTUAL response conveys the same meaning and information as the EXPECTED response. Minor differences in wording are acceptable as long as the core meaning is preserved.

Respond with a JSON object:
{{
    "score": <float from 0.0 to 1.0>,
    "reasoning": "<brief explanation of your evaluation>"
}}

Score guidelines:
- 1.0: Perfect semantic match, all key information preserved
- 0.8-0.9: Minor differences but core meaning intact
- 0.5-0.7: Partial match, some key information missing or different
- 0.2-0.4: Significant differences but some overlap
- 0.0-0.1: Completely different or contradictory
"""

QUALITY_EVALUATION_PROMPT = """You are an expert evaluator. Your task is to evaluate the quality of an AI assistant's response.

QUESTION: {question}

RESPONSE: {response}

{rubric_section}

Evaluate the response and provide a score.

Respond with a JSON object:
{{
    "score": <float from 0.0 to 1.0>,
    "reasoning": "<brief explanation of your evaluation>"
}}
"""


class LLMJudgeCriterion(BaseCriterion):
    """Use an LLM to judge response quality and semantic match.

    This criterion sends the actual and expected responses to an LLM
    for evaluation, enabling semantic comparison that goes beyond
    simple text matching.

    Supports majority voting across multiple samples for more reliable
    evaluation.

    Attributes:
        name: "final_response_match_v2"
        judge_model: The model to use for evaluation
        num_samples: Number of samples for majority voting

    Example:
        ```python
        criterion = LLMJudgeCriterion(
            config=CriterionConfig(
                threshold=0.8,
                judge_model="gpt-4o-mini",
                num_samples=3,
            )
        )

        result = await criterion.evaluate(collector, eval_case)
        print(f"LLM judge score: {result.score}")
        ```
    """

    name = "final_response_match_v2"
    description = "LLM-based semantic response matching"

    async def evaluate(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
    ) -> CriterionResult:
        """Evaluate using LLM as judge.

        Args:
            actual: Collected trajectory from actual execution.
            expected: The evaluation case with expected outcomes.

        Returns:
            CriterionResult with LLM-judged score.
        """
        try:
            # Extract question and responses
            question = self._extract_question(expected)
            expected_response = self._extract_expected_response(expected)
            actual_response = self._extract_actual_response(actual)

            if not expected_response:
                return CriterionResult.success(
                    criterion=self.name,
                    score=1.0,
                    threshold=self.threshold,
                    details={"note": "No expected response to compare"},
                )

            # Call LLM for evaluation
            scores, reasonings = await self._evaluate_with_llm(
                question=question,
                expected=expected_response,
                actual=actual_response,
            )

            # Compute final score (average of samples)
            final_score = sum(scores) / len(scores) if scores else 0.0

            return CriterionResult.success(
                criterion=self.name,
                score=final_score,
                threshold=self.threshold,
                details={
                    "samples": len(scores),
                    "individual_scores": scores,
                    "reasonings": reasonings,
                    "judge_model": self.config.judge_model,
                },
            )

        except Exception as e:
            logger.error("LLM judge evaluation failed: %s", e)
            return CriterionResult.failure(
                criterion=self.name,
                error=str(e),
            )

    async def _evaluate_with_llm(
        self,
        question: str,
        expected: str,
        actual: str,
    ) -> tuple[list[float], list[str]]:
        """Call LLM for evaluation.

        Args:
            question: The original question/query.
            expected: Expected response text.
            actual: Actual response text.

        Returns:
            Tuple of (scores, reasonings) from each sample.
        """
        scores = []
        reasonings = []

        prompt = SEMANTIC_MATCH_PROMPT.format(
            question=question,
            expected=expected,
            actual=actual,
        )

        num_samples = self.config.num_samples

        for _ in range(num_samples):
            try:
                score, reasoning = await self._call_llm(prompt)
                scores.append(score)
                reasonings.append(reasoning)
            except Exception as e:
                logger.warning("LLM sample failed: %s", e)
                # Continue with other samples

        return scores, reasonings

    async def _call_llm(self, prompt: str) -> tuple[float, str]:
        """Call the LLM and parse the response.

        Args:
            prompt: The evaluation prompt.

        Returns:
            Tuple of (score, reasoning).
        """
        import json

        try:
            # Try using litellm if available
            from litellm import acompletion

            response = await acompletion(
                model=self.config.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            content = response.choices[0].message.content
            # Parse JSON response
            result = json.loads(content)
            return float(result.get("score", 0.0)), result.get("reasoning", "")

        except ImportError:
            # Fallback - try OpenAI directly
            try:
                from openai import AsyncOpenAI

                client = AsyncOpenAI()
                response = await client.chat.completions.create(
                    model=self.config.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )

                content = response.choices[0].message.content
                result = json.loads(content)
                return float(result.get("score", 0.0)), result.get("reasoning", "")

            except ImportError:
                logger.warning("No LLM library available, returning mock score")
                return 0.5, "No LLM available for evaluation"

    def _extract_question(self, expected: EvalCase) -> str:
        """Extract the user question from the eval case."""
        if expected.conversation:
            return expected.conversation[0].user_content.get_text()
        return ""

    def _extract_expected_response(self, expected: EvalCase) -> str:
        """Extract expected response from eval case."""
        for invocation in expected.conversation:
            if invocation.expected_final_response:
                return invocation.expected_final_response.get_text()
        return ""

    def _extract_actual_response(self, collector: TrajectoryCollector) -> str:
        """Extract actual response from collector."""
        if collector.messages:
            for msg in reversed(collector.messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
        return ""


class RubricBasedCriterion(BaseCriterion):
    """Evaluate response quality against custom rubrics.

    Uses an LLM to grade responses against user-defined rubrics,
    allowing for flexible and domain-specific evaluation.

    Attributes:
        name: "rubric_based_final_response_quality_v1"
    """

    name = "rubric_based_final_response_quality_v1"
    description = "LLM-based rubric evaluation"

    async def evaluate(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
    ) -> CriterionResult:
        """Evaluate against rubrics.

        Args:
            actual: Collected trajectory from actual execution.
            expected: The evaluation case with expected outcomes.

        Returns:
            CriterionResult with rubric-based score.
        """
        try:
            rubrics = self.config.rubrics
            if not rubrics:
                return CriterionResult.success(
                    criterion=self.name,
                    score=1.0,
                    threshold=self.threshold,
                    details={"note": "No rubrics configured"},
                )

            question = self._extract_question(expected)
            actual_response = self._extract_actual_response(actual)

            # Build rubric section
            rubric_lines = ["Evaluate the response against these criteria:"]
            for rubric in rubrics:
                rubric_lines.append(f"- {rubric.rubric_id}: {rubric.content}")
            rubric_section = "\n".join(rubric_lines)

            prompt = QUALITY_EVALUATION_PROMPT.format(
                question=question,
                response=actual_response,
                rubric_section=rubric_section,
            )

            scores = []
            reasonings = []

            for _ in range(self.config.num_samples):
                try:
                    score, reasoning = await self._call_llm(prompt)
                    scores.append(score)
                    reasonings.append(reasoning)
                except Exception as e:
                    logger.warning("Rubric sample failed: %s", e)

            final_score = sum(scores) / len(scores) if scores else 0.0

            return CriterionResult.success(
                criterion=self.name,
                score=final_score,
                threshold=self.threshold,
                details={
                    "rubrics": [r.rubric_id for r in rubrics],
                    "scores": scores,
                    "reasonings": reasonings,
                },
            )

        except Exception as e:
            logger.error("Rubric evaluation failed: %s", e)
            return CriterionResult.failure(
                criterion=self.name,
                error=str(e),
            )

    async def _call_llm(self, prompt: str) -> tuple[float, str]:
        """Call the LLM and parse response."""
        import json

        try:
            from litellm import acompletion

            response = await acompletion(
                model=self.config.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )

            content = response.choices[0].message.content
            result = json.loads(content)
            return float(result.get("score", 0.0)), result.get("reasoning", "")

        except ImportError:
            return 0.5, "No LLM available"

    def _extract_question(self, expected: EvalCase) -> str:
        """Extract question from eval case."""
        if expected.conversation:
            return expected.conversation[0].user_content.get_text()
        return ""

    def _extract_actual_response(self, collector: TrajectoryCollector) -> str:
        """Extract actual response."""
        if collector.messages:
            for msg in reversed(collector.messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
        return ""
