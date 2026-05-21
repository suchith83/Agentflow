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

from pydantic import BaseModel, Field, field_validator


# Default judge model used across all LLM-based evaluation criteria.
# Users can override per-criterion or per-preset by passing judge_model="gpt-4o" etc.
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"


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
        keywords: Required keywords for ContainsKeywordsCriterion.
        check_args: Whether to check tool arguments in trajectory matching.
        enabled: Whether this criterion is enabled.
    """

    threshold: float = 0.8
    match_type: MatchType = MatchType.EXACT
    judge_model: str = DEFAULT_JUDGE_MODEL
    num_samples: int = 3
    rubrics: list[Rubric] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    check_args: bool = False
    enabled: bool = True

    @classmethod
    def tool_name_match(cls, threshold: float = 1.0) -> CriterionConfig:
        """Create configuration for tool name matching (no LLM).

        Checks that the names of tools called match the expected list.
        """
        return cls(threshold=threshold)

    @classmethod
    def trajectory(
        cls,
        threshold: float = 1.0,
        match_type: MatchType = MatchType.EXACT,
        check_args: bool = False,
    ) -> CriterionConfig:
        """Create configuration for tool sequence matching (no LLM).

        Use match_type to control strictness:
        - EXACT: same tools, same order, no extras
        - IN_ORDER: expected tools appear in order, extras allowed
        - ANY_ORDER: expected tools appear in any order, extras allowed
        """
        return cls(
            threshold=threshold,
            match_type=match_type,
            check_args=check_args,
        )

    @classmethod
    def node_order(
        cls,
        threshold: float = 1.0,
        match_type: MatchType = MatchType.EXACT,
    ) -> CriterionConfig:
        """Create configuration for node visit order matching (no LLM).

        Checks that the graph visited nodes in the expected order.
        Use match_type to control strictness:
        - EXACT: same nodes, same order, same count
        - IN_ORDER: expected nodes appear in order, extras allowed
        - ANY_ORDER: expected nodes all present, any order
        """
        return cls(threshold=threshold, match_type=match_type)

    @classmethod
    def response_match(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for LLM-based semantic response matching.

        Uses an LLM to judge whether the actual response is semantically
        equivalent to the expected response.  Handles paraphrasing and
        differently-worded but correct answers.

        For a fast, free (no LLM) alternative use :meth:`rouge_match`.
        """
        return cls(threshold=threshold, judge_model=judge_model, num_samples=num_samples)

    @classmethod
    def rouge_match(cls, threshold: float = 0.5) -> CriterionConfig:
        """Create configuration for ROUGE-1 F1 response similarity (no LLM).

        Uses token-overlap (ROUGE-1 F1) to measure similarity between the
        actual and expected response.  No API calls — fast and free.

        Use with criterion key ``"rouge_match"`` in :class:`EvalConfig`.

        For semantic/paraphrase-aware matching use :meth:`response_match`
        (LLM-based) instead.
        """
        return cls(threshold=threshold)

    @classmethod
    def contains_keywords(
        cls,
        keywords: list[str],
        threshold: float = 1.0,
    ) -> CriterionConfig:
        """Create configuration for keyword presence check (no LLM).

        Args:
            keywords: List of keywords that must appear in the actual response.
            threshold: Fraction of keywords that must be present (0.0 to 1.0).
        """
        return cls(threshold=threshold, keywords=keywords)

    @classmethod
    def llm_judge(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for LLM-as-judge overall quality scoring."""
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
        judge_model: str = DEFAULT_JUDGE_MODEL,
    ) -> CriterionConfig:
        """Create configuration for custom rubric-based scoring."""
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            rubrics=rubrics,
        )

    @classmethod
    def factual_accuracy(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for factual accuracy evaluation.

        Checks whether all stated facts in the response are correct —
        numbers, dates, names, and verifiable claims.
        """
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            num_samples=num_samples,
        )

    @classmethod
    def hallucination(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for hallucination/groundedness detection.

        Checks whether the response is grounded in the context the agent
        actually had (tool results, provided facts). Score 1.0 = fully
        grounded, 0.0 = mostly hallucinated.
        """
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            num_samples=num_samples,
        )

    @classmethod
    def safety(
        cls,
        threshold: float = 0.8,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 3,
    ) -> CriterionConfig:
        """Create configuration for safety evaluation.

        Scores the response across five dimensions:
        harmful_content, hate_speech, privacy, misinformation, manipulation.
        The overall score is the minimum across all categories — one unsafe
        category fails the criterion.
        """
        return cls(
            threshold=threshold,
            judge_model=judge_model,
            num_samples=num_samples,
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

    model: str = "gemini-2.5-flash"
    max_invocations: int = 10
    temperature: float = 0.7
    thinking_enabled: bool = False
    thinking_budget: int = 10240


class ReporterConfig(BaseModel):
    """Configuration for evaluation result reporters.

    Controls which reporters are enabled and where output files
    are written after an evaluation completes.

    Attributes:
        enabled: Master switch — when False, no reporters run automatically.
        output_dir: Directory for generated report files.
        console: Enable console (stdout) reporting.
        json_report: Enable JSON file reporting.
        html: Enable HTML file reporting.
        junit_xml: Enable JUnit XML file reporting.
        verbose: Verbose console output (show all cases, not just failures).
        include_details: Include full criterion details in file reports.
        include_trajectory: Include trajectory data in JSON reports.
        include_node_responses: Include per-node intermediate data in reports.
        include_actual_response: Include agent final response in reports.
        include_tool_call_details: Include tool arguments and results in reports.
        timestamp_files: Append timestamp to generated filenames.
    """

    enabled: bool = True
    output_dir: str = "eval_reports"
    console: bool = True
    json_report: bool = True
    html: bool = True
    junit_xml: bool = False
    verbose: bool = True
    include_details: bool = True
    include_trajectory: bool = True
    include_node_responses: bool = True
    include_actual_response: bool = True
    include_tool_call_details: bool = True
    timestamp_files: bool = True


class CriteriaConfig(BaseModel):
    """Type-safe configuration for evaluation criteria.

    This replaces dictionary-based criterion configuration with a typed
    Pydantic model that has explicit fields for each criterion type.
    This prevents mistyping criterion names and provides IDE autocomplete.

    All fields are optional — only include the criteria you need.

    Attributes:
        tool_name_match: Configuration for tool name matching criterion.
        trajectory: Configuration for tool trajectory matching criterion.
        node_order: Configuration for node visit order matching criterion.
        response_match: Configuration for response semantic matching criterion (LLM-based).
        rouge_match: Configuration for ROUGE-1 F1 response similarity criterion.
        contains_keywords: Configuration for keyword presence criterion.
        llm_judge: Configuration for LLM-as-judge overall quality criterion.
        rubric_based: Configuration for custom rubric-based criterion.
        factual_accuracy: Configuration for factual accuracy criterion (LLM-based).
        hallucination: Configuration for hallucination/groundedness criterion (LLM-based).
        safety: Configuration for safety evaluation criterion (LLM-based).
        simulation_goals: Configuration for simulation goals criterion.
    """

    model_config = {"extra": "forbid"}  # Reject extra fields for type safety

    tool_name_match: CriterionConfig | None = None
    trajectory: CriterionConfig | None = None
    node_order: CriterionConfig | None = None
    response_match: CriterionConfig | None = None
    rouge_match: CriterionConfig | None = None
    contains_keywords: CriterionConfig | None = None
    llm_judge: CriterionConfig | None = None
    rubric_based: CriterionConfig | None = None
    factual_accuracy: CriterionConfig | None = None
    hallucination: CriterionConfig | None = None
    safety: CriterionConfig | None = None
    simulation_goals: CriterionConfig | None = None

    def to_dict(self) -> dict[str, CriterionConfig]:
        """Convert typed criteria to dictionary format for backward compatibility.

        Returns:
            Dictionary mapping criterion names to configurations.
        """
        result = {}
        field_mapping = {
            "tool_name_match": "tool_name_match_score",
            "trajectory": "tool_trajectory_avg_score",
            "node_order": "node_order",
            "response_match": "response_match_score",
            "rouge_match": "rouge_match",
            "contains_keywords": "contains_keywords",
            "llm_judge": "final_response_match_v2",
            "rubric_based": "rubric_based_score",
            "factual_accuracy": "factual_accuracy_score",
            "hallucination": "hallucinations_v1",
            "safety": "safety_score",
            "simulation_goals": "simulation_goals_match",
        }

        for field_name, criterion_name in field_mapping.items():
            value = getattr(self, field_name)
            if value is not None:
                result[criterion_name] = value

        return result

    # --- dict-compatible helpers so consumers like AgentEvaluator keep working ---

    def items(self):
        """Yield (criterion_name, CriterionConfig) pairs — dict-compatible."""
        return self.to_dict().items()

    def keys(self):
        """Return criterion names — dict-compatible."""
        return self.to_dict().keys()

    def values(self):
        """Return CriterionConfig instances — dict-compatible."""
        return self.to_dict().values()

    def get(self, name: str, default=None):
        """Return the CriterionConfig for *name* — dict-compatible."""
        return self.to_dict().get(name, default)

    def __getitem__(self, name: str):
        result = self.to_dict()
        return result[name]

    def __contains__(self, name: str):
        return name in self.to_dict()

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self):
        return len(self.to_dict())

    @classmethod
    def from_dict(cls, criteria_dict: dict[str, CriterionConfig]) -> CriteriaConfig:
        """Create typed CriteriaConfig from dictionary format.

        Args:
            criteria_dict: Dictionary mapping criterion names to configurations.

        Returns:
            CriteriaConfig instance.
        """
        reverse_mapping = {
            "tool_name_match_score": "tool_name_match",
            "tool_trajectory_avg_score": "trajectory",
            "node_order": "node_order",
            "response_match_score": "response_match",
            "rouge_match": "rouge_match",
            "contains_keywords": "contains_keywords",
            "final_response_match_v2": "llm_judge",
            "rubric_based_score": "rubric_based",
            "factual_accuracy_score": "factual_accuracy",
            "hallucinations_v1": "hallucination",
            "safety_score": "safety",
            "simulation_goals_match": "simulation_goals",
        }

        data = {}
        for criterion_name, config in criteria_dict.items():
            field_name = reverse_mapping.get(criterion_name)
            if field_name:
                data[field_name] = config

        return cls(**data)


class EvalConfig(BaseModel):
    """Main evaluation configuration.

    Contains all settings for running an evaluation, including
    which criteria to use and their thresholds.

    Attributes:
        criteria: Type-safe criteria configuration (CriteriaConfig instance).
        user_simulator_config: Configuration for user simulation.
        parallel: Whether to run evaluations in parallel.
        max_concurrency: Maximum concurrent evaluations if parallel.
        timeout: Timeout for each evaluation case (seconds).
        verbose: Whether to output verbose logging.
        mock_mode: Whether to run in mock mode (no actual execution).
        reporter: Configuration for automatic report generation.
    """

    criteria: CriteriaConfig = Field(default_factory=CriteriaConfig)
    user_simulator_config: UserSimulatorConfig | None = None
    parallel: bool = False
    max_concurrency: int = 4
    timeout: float = 300.0
    verbose: bool = False
    mock_mode: bool = False
    reporter: ReporterConfig = Field(default_factory=ReporterConfig)

    @field_validator("criteria", mode="before")
    @classmethod
    def _normalize_criteria(cls, value):
        if isinstance(value, dict):
            return CriteriaConfig.from_dict(value)
        return value

    @classmethod
    def default(cls) -> EvalConfig:
        """Create default evaluation configuration.

        Default includes:
            - tool_trajectory_avg_score: EXACT match, threshold 1.0
            - response_match_score: ROUGE-1, threshold 0.8
        """
        return cls(
            criteria=CriteriaConfig(
                trajectory=CriterionConfig.trajectory(
                    threshold=1.0,
                    match_type=MatchType.EXACT,
                ),
                response_match=CriterionConfig.response_match(
                    threshold=0.8,
                ),
            )
        )

    @classmethod
    def strict(cls) -> EvalConfig:
        """Create strict evaluation configuration.

        All criteria set to maximum strictness.
        """
        return cls(
            criteria=CriteriaConfig(
                trajectory=CriterionConfig.trajectory(
                    threshold=1.0,
                    match_type=MatchType.EXACT,
                    check_args=True,
                ),
                response_match=CriterionConfig.response_match(
                    threshold=0.9,
                ),
                llm_judge=CriterionConfig.llm_judge(
                    threshold=0.9,
                    num_samples=5,
                ),
            )
        )

    @classmethod
    def relaxed(cls) -> EvalConfig:
        """Create relaxed evaluation configuration.

        Uses IN_ORDER matching and lower thresholds.
        """
        return cls(
            criteria=CriteriaConfig(
                trajectory=CriterionConfig.trajectory(
                    threshold=0.8,
                    match_type=MatchType.IN_ORDER,
                    check_args=False,
                ),
                response_match=CriterionConfig.response_match(
                    threshold=0.6,
                ),
            )
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
        new_config.criteria.rubric_based = CriterionConfig.rubric_based(
            rubrics=rubrics,
        )
        return new_config
