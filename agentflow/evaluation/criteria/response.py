"""
Response matching criterion.

Compares actual responses against expected responses using various
text similarity metrics, primarily ROUGE-1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentflow.evaluation.criteria.base import SyncCriterion
from agentflow.evaluation.eval_result import CriterionResult


if TYPE_CHECKING:
    from agentflow.evaluation.collectors.trajectory_collector import TrajectoryCollector
    from agentflow.evaluation.eval_set import EvalCase


class ResponseMatchCriterion(SyncCriterion):
    """Compare response similarity using ROUGE-1 F1 score.

    ROUGE-1 measures unigram overlap between the actual and expected
    responses, providing a simple but effective text similarity metric.

    Attributes:
        name: "response_match_score"

    Example:
        ```python
        criterion = ResponseMatchCriterion(config=CriterionConfig(threshold=0.7))

        result = await criterion.evaluate(collector, eval_case)
        print(f"Response match: {result.score:.2%}")
        ```
    """

    name = "response_match_score"
    description = "Evaluates response text similarity using ROUGE-1"

    def evaluate_sync(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
    ) -> CriterionResult:
        """Evaluate response match.

        Args:
            actual: Collected trajectory from actual execution.
            expected: The evaluation case with expected outcomes.

        Returns:
            CriterionResult with ROUGE-1 F1 score.
        """
        # Get expected response from the last invocation
        expected_response = ""
        for invocation in expected.conversation:
            if invocation.expected_final_response:
                expected_response = invocation.expected_final_response.get_text()

        # Get actual response from collector
        actual_response = self._extract_actual_response(actual)

        if not expected_response:
            # No expected response defined - pass if no actual response or give partial credit
            score = 1.0 if not actual_response else 0.5
            return CriterionResult.success(
                criterion=self.name,
                score=score,
                threshold=self.threshold,
                details={"note": "No expected response defined"},
            )

        # Calculate ROUGE-1 F1 score
        precision, recall, f1 = self._rouge1_score(actual_response, expected_response)

        return CriterionResult.success(
            criterion=self.name,
            score=f1,
            threshold=self.threshold,
            details={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "actual_response": actual_response[:500],  # Truncate for readability
                "expected_response": expected_response[:500],
            },
        )

    def _extract_actual_response(self, collector: TrajectoryCollector) -> str:
        """Extract the actual response from the collector.

        This looks through captured messages for the final assistant response.
        """
        # Check if collector has captured messages
        if collector.messages:
            for msg in reversed(collector.messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        # Handle content blocks
                        texts = []
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                texts.append(block["text"])
                        return " ".join(texts)
        return ""

    def _rouge1_score(
        self,
        actual: str,
        expected: str,
    ) -> tuple[float, float, float]:
        """Calculate ROUGE-1 precision, recall, and F1.

        Args:
            actual: The actual response text.
            expected: The expected response text.

        Returns:
            Tuple of (precision, recall, f1).
        """
        # Tokenize by whitespace and lowercase
        actual_tokens = set(self._tokenize(actual))
        expected_tokens = set(self._tokenize(expected))

        if not expected_tokens:
            return (1.0, 1.0, 1.0) if not actual_tokens else (0.0, 0.0, 0.0)

        if not actual_tokens:
            return (0.0, 0.0, 0.0)

        # Calculate overlap
        overlap = actual_tokens & expected_tokens
        overlap_count = len(overlap)

        precision = overlap_count / len(actual_tokens)
        recall = overlap_count / len(expected_tokens)

        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        return (precision, recall, f1)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenization with lowercasing.

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        # Remove punctuation and lowercase
        import re

        text = re.sub(r"[^\w\s]", " ", text.lower())
        return [t for t in text.split() if t]


class ExactMatchCriterion(SyncCriterion):
    """Check for exact string match between actual and expected response.

    This is a stricter criterion than ROUGE-1, requiring identical text.
    Useful for highly deterministic responses.
    """

    name = "exact_response_match"
    description = "Evaluates exact string match of response"

    def evaluate_sync(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
    ) -> CriterionResult:
        """Evaluate exact match."""
        # Get expected response
        expected_response = ""
        for invocation in expected.conversation:
            if invocation.expected_final_response:
                expected_response = invocation.expected_final_response.get_text().strip()

        # Get actual response - need to extract from collector
        actual_response = ""
        if actual.messages:
            for msg in reversed(actual.messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        actual_response = content.strip()
                        break

        # Exact match
        is_match = actual_response == expected_response
        score = 1.0 if is_match else 0.0

        return CriterionResult.success(
            criterion=self.name,
            score=score,
            threshold=self.threshold,
            details={
                "match": is_match,
                "actual_length": len(actual_response),
                "expected_length": len(expected_response),
            },
        )


class ContainsKeywordsCriterion(SyncCriterion):
    """Check if response contains required keywords.

    Useful for checking that specific information is included in responses.
    """

    name = "contains_keywords"
    description = "Evaluates presence of required keywords in response"

    def __init__(
        self,
        keywords: list[str] | None = None,
        config=None,
    ):
        """Initialize with required keywords.

        Args:
            keywords: List of keywords that must appear in the response.
            config: Optional criterion configuration.
        """
        super().__init__(config)
        self.keywords = keywords or []

    def evaluate_sync(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
    ) -> CriterionResult:
        """Evaluate keyword presence."""
        # Get actual response
        actual_response = ""
        if actual.messages:
            for msg in reversed(actual.messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        actual_response = content.lower()
                        break

        if not self.keywords:
            return CriterionResult.success(
                criterion=self.name,
                score=1.0,
                threshold=self.threshold,
                details={"note": "No keywords specified"},
            )

        # Check each keyword
        found = []
        missing = []
        for keyword in self.keywords:
            if keyword.lower() in actual_response:
                found.append(keyword)
            else:
                missing.append(keyword)

        score = len(found) / len(self.keywords)

        return CriterionResult.success(
            criterion=self.name,
            score=score,
            threshold=self.threshold,
            details={
                "found": found,
                "missing": missing,
                "keywords": self.keywords,
            },
        )
