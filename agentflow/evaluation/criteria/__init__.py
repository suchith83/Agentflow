"""
Evaluation criteria for agent assessment.

This module provides various criteria for evaluating agent behavior:
    - BaseCriterion: Abstract base class for all criteria
    - SyncCriterion: Base class for synchronous criteria
    - CompositeCriterion: Combine multiple criteria with AND/OR logic
    - WeightedCriterion: Weighted average of multiple criteria
    - TrajectoryMatchCriterion: Tool trajectory matching (EXACT, IN_ORDER, ANY_ORDER)
    - ResponseMatchCriterion: Response similarity using ROUGE-1
    - LLMJudgeCriterion: LLM-as-judge semantic evaluation
    - HallucinationCriterion: Groundedness/hallucination detection
    - SafetyCriterion: Safety and harmlessness evaluation
    - FactualAccuracyCriterion: Factual accuracy checking
"""

from agentflow.evaluation.criteria.base import (
    BaseCriterion,
    CompositeCriterion,
    SyncCriterion,
    WeightedCriterion,
)
from agentflow.evaluation.criteria.llm_judge import (
    LLMJudgeCriterion,
    RubricBasedCriterion,
)
from agentflow.evaluation.criteria.response import (
    ContainsKeywordsCriterion,
    ExactMatchCriterion,
    ResponseMatchCriterion,
)
from agentflow.evaluation.criteria.trajectory import (
    ToolNameMatchCriterion,
    TrajectoryMatchCriterion,
)
from agentflow.evaluation.criteria.advanced import (
    HallucinationCriterion,
    SafetyCriterion,
    FactualAccuracyCriterion,
)

__all__ = [
    # Base classes
    "BaseCriterion",
    "CompositeCriterion",
    "SyncCriterion",
    "WeightedCriterion",
    # Trajectory criteria
    "TrajectoryMatchCriterion",
    "ToolNameMatchCriterion",
    # Response criteria
    "ResponseMatchCriterion",
    "ExactMatchCriterion",
    "ContainsKeywordsCriterion",
    # LLM-as-judge criteria
    "LLMJudgeCriterion",
    "RubricBasedCriterion",
    # Advanced criteria
    "HallucinationCriterion",
    "SafetyCriterion",
    "FactualAccuracyCriterion",
]
