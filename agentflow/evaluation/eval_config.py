"""
Evaluation configuration models.

This module defines configuration structures for agent evaluation:
    - EvalConfig: Main configuration container
    - CriterionConfig: Configuration for individual criteria
    - Rubric: Custom rubric definition
    - UserSimulatorConfig: Configuration for user simulation
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MatchType(str, Enum):
    """Match type for trajectory comparison.

    Values:
        EXACT: Require perfect match - same tools, args, and order.
        IN_ORDER: Expected tools must appear in order, extras allowed.
        ANY_ORDER: Expected tools must appear in any order, extras allowed.
    """

    EXACT = "EXACT"
    IN_ORDER = "IN_ORDER"
    ANY_ORDER = "ANY_ORDER"


class Rubric(BaseModel):
    """A custom evaluation rubric.

    Rubrics define specific criteria for evaluating agent behavior
    using LLM-as-judge evaluation.

    Attributes:
        rubric_id: Unique identifier for this rubric.
        content: The rubric description/criteria text.
        weight: Weight of this rubric in overall scoring (default 1.0).
    """

    rubric_id: str
    content: str
    weight: float = 1.0

    @classmethod
    def create(cls, rubric_id: str, content: str, weight: float = 1.0) -> Rubric:
        """Create a rubric with the given parameters."""
        return cls(rubric_id=rubric_id, content=content, weight=weight)


class CriterionConfig(BaseModel):
    """Configuration for a single evaluation criterion.

    Attributes:
        threshold: Minimum score to pass (0.0 to 1.0).
        match_type: Match type for trajectory criteria.
        judge_model: Model to use for LLM-as-judge criteria.
        num_samples: Number of samples for LLM judge (majority vote).
        rubrics: List of custom rubrics for rubric-based criteria.
        check_args: Whether to check tool arguments in trajectory matching.
        enabled: Whether this criterion is enabled.
    """

    threshold: float = 0.8
    match_type: MatchType = MatchType.EXACT
    judge_model: str = "gpt-4o-mini"
    num_samples: int = 3
    rubrics: list[Rubric] = Field(default_factory=list)
    check_args: bool = True
    enabled: bool = True

    @classmethod
    def trajectory(
        cls,
        threshold: float = 1.0,
        match_type: MatchType = MatchType.EXACT,
        check_args: bool = True,
    ) -> CriterionConfig:
        """Create configuration for trajectory matching."""
        return cls(
            threshold=threshold,
            match_type=match_type,
            check_args=check_args,
        )

    @classmethod
    def response_match(cls, threshold: float = 0.7) -> CriterionConfig:
        """Create configuration for response matching (ROUGE-1)."""
        return cls(threshold=threshold)

    @classmethod
    def llm_judge(
        cls,
        threshold: float = 0.8,
        judge_model: str = "gpt-4o-mini",
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for LLM-as-judge evaluation."""
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            num_samples=num_samples,
        )

    @classmethod
    def rubric_based(
        cls,
        rubrics: list[Rubric],
        threshold: float = 0.8,
        judge_model: str = "gpt-4o-mini",
    ) -> CriterionConfig:
        """Create configuration for rubric-based evaluation."""
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            rubrics=rubrics,
        )


class UserSimulatorConfig(BaseModel):
    """Configuration for AI-powered user simulation.

    Attributes:
        model: Model to use for generating user prompts.
        max_invocations: Maximum number of conversation turns.
        temperature: Temperature for generation.
        thinking_enabled: Whether to enable thinking/reasoning.
        thinking_budget: Token budget for thinking (if enabled).
    """

    model: str = "gpt-4o"
    max_invocations: int = 10
    temperature: float = 0.7
    thinking_enabled: bool = False
    thinking_budget: int = 10240


class EvalConfig(BaseModel):
    """Main evaluation configuration.

    Contains all settings for running an evaluation, including
    which criteria to use and their thresholds.

    Attributes:
        criteria: Dictionary of criterion name to configuration.
        user_simulator_config: Configuration for user simulation.
        parallel: Whether to run evaluations in parallel.
        max_concurrency: Maximum concurrent evaluations if parallel.
        timeout: Timeout for each evaluation case (seconds).
        verbose: Whether to output verbose logging.
        mock_mode: Whether to run in mock mode (no actual execution).
    """

    criteria: dict[str, CriterionConfig] = Field(default_factory=dict)
    user_simulator_config: UserSimulatorConfig | None = None
    parallel: bool = False
    max_concurrency: int = 4
    timeout: float = 300.0
    verbose: bool = False
    mock_mode: bool = False

    @classmethod
    def default(cls) -> EvalConfig:
        """Create default evaluation configuration.

        Default includes:
            - tool_trajectory_avg_score: EXACT match, threshold 1.0
            - response_match_score: ROUGE-1, threshold 0.8
        """
        return cls(
            criteria={
                "tool_trajectory_avg_score": CriterionConfig.trajectory(
                    threshold=1.0,
                    match_type=MatchType.EXACT,
                ),
                "response_match_score": CriterionConfig.response_match(
                    threshold=0.8,
                ),
            }
        )

    @classmethod
    def strict(cls) -> EvalConfig:
        """Create strict evaluation configuration.

        All criteria set to maximum strictness.
        """
        return cls(
            criteria={
                "tool_trajectory_avg_score": CriterionConfig.trajectory(
                    threshold=1.0,
                    match_type=MatchType.EXACT,
                    check_args=True,
                ),
                "response_match_score": CriterionConfig.response_match(
                    threshold=0.9,
                ),
                "final_response_match_v2": CriterionConfig.llm_judge(
                    threshold=0.9,
                    num_samples=5,
                ),
            }
        )

    @classmethod
    def relaxed(cls) -> EvalConfig:
        """Create relaxed evaluation configuration.

        Uses IN_ORDER matching and lower thresholds.
        """
        return cls(
            criteria={
                "tool_trajectory_avg_score": CriterionConfig.trajectory(
                    threshold=0.8,
                    match_type=MatchType.IN_ORDER,
                    check_args=False,
                ),
                "response_match_score": CriterionConfig.response_match(
                    threshold=0.6,
                ),
            }
        )

    @classmethod
    def from_file(cls, path: str) -> EvalConfig:
        """Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            Loaded EvalConfig instance.
        """
        import json
        from pathlib import Path

        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_file(self, path: str) -> None:
        """Save configuration to a JSON file.

        Args:
            path: Path to save the JSON file.
        """
        import json
        from pathlib import Path

        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    def get_criterion_config(self, name: str) -> CriterionConfig | None:
        """Get configuration for a specific criterion."""
        return self.criteria.get(name)

    def enable_criterion(
        self,
        name: str,
        config: CriterionConfig | None = None,
    ) -> None:
        """Enable a criterion with optional configuration."""
        if config:
            self.criteria[name] = config
        elif name not in self.criteria:
            self.criteria[name] = CriterionConfig()

    def disable_criterion(self, name: str) -> None:
        """Disable a criterion."""
        if name in self.criteria:
            self.criteria[name].enabled = False

    def with_rubrics(self, rubrics: list[Rubric]) -> EvalConfig:
        """Return a copy with rubric-based criteria added."""
        import copy

        new_config = copy.deepcopy(self)
        new_config.criteria["rubric_based_quality"] = CriterionConfig.rubric_based(
            rubrics=rubrics,
        )
        return new_config
