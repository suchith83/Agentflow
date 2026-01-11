"""
Agent Evaluation Module for 10xScale Agentflow.

This module provides comprehensive evaluation capabilities for agent graphs,
including trajectory analysis, response quality assessment, and LLM-as-judge
evaluation patterns.

Main Components:
    - AgentEvaluator: Main class for running evaluations
    - EvalSet, EvalCase: Data models for test cases
    - EvalConfig: Configuration for evaluation criteria
    - TrajectoryCollector: Captures execution trajectory from events
    - BaseCriterion: Base class for evaluation criteria
    - Reporters: Console, JSON, HTML report generators
    - UserSimulator: AI-powered conversation simulation

Example:
    ```python
    from agentflow.evaluation import AgentEvaluator, EvalConfig

    evaluator = AgentEvaluator(graph, config=EvalConfig.default())
    report = await evaluator.evaluate("tests/fixtures/my_tests.evalset.json")
    print(report.summary)
    ```
"""

from agentflow.evaluation.builder import EvalSetBuilder
from agentflow.evaluation.collectors.trajectory_collector import (
    EventCollector,
    TrajectoryCollector,
)
from agentflow.evaluation.criteria.advanced import (
    FactualAccuracyCriterion,
    HallucinationCriterion,
    SafetyCriterion,
)
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
from agentflow.evaluation.eval_config import (
    CriterionConfig,
    EvalConfig,
    MatchType,
    Rubric,
    UserSimulatorConfig,
)
from agentflow.evaluation.eval_result import (
    CriterionResult,
    EvalCaseResult,
    EvalReport,
    EvalSummary,
)
from agentflow.evaluation.eval_set import (
    EvalCase,
    EvalSet,
    Invocation,
    MessageContent,
    SessionInput,
    StepType,
    ToolCall,
    TrajectoryStep,
)
from agentflow.evaluation.evaluator import (
    AgentEvaluator,
    EvaluationRunner,
)
from agentflow.evaluation.presets import EvalPresets
from agentflow.evaluation.quick_eval import QuickEval
from agentflow.evaluation.reporters.console import (
    ConsoleReporter,
    print_report,
)
from agentflow.evaluation.reporters.html import (
    HTMLReporter,
)
from agentflow.evaluation.reporters.json import (
    JSONReporter,
    JUnitXMLReporter,
)
from agentflow.evaluation.simulators.user_simulator import (
    BatchSimulator,
    ConversationScenario,
    SimulationResult,
    UserSimulator,
)


__all__ = [
    # Evaluator
    "AgentEvaluator",
    # Base Criteria
    "BaseCriterion",
    "BatchSimulator",
    "CompositeCriterion",
    # Reporters
    "ConsoleReporter",
    "ContainsKeywordsCriterion",
    "ConversationScenario",
    # Configuration
    "CriterionConfig",
    # Results
    "CriterionResult",
    # Data models - eval_set
    "EvalCase",
    "EvalCaseResult",
    "EvalConfig",
    # New: Simplified interfaces
    "EvalPresets",
    "EvalReport",
    "EvalSet",
    "EvalSetBuilder",
    "EvalSummary",
    "EvaluationRunner",
    # Collectors
    "EventCollector",
    "ExactMatchCriterion",
    "FactualAccuracyCriterion",
    "HTMLReporter",
    # Advanced Criteria
    "HallucinationCriterion",
    "Invocation",
    "JSONReporter",
    "JUnitXMLReporter",
    # LLM-as-Judge Criteria
    "LLMJudgeCriterion",
    "MatchType",
    "MessageContent",
    # New: Quick evaluation
    "QuickEval",
    # Response Criteria
    "ResponseMatchCriterion",
    "Rubric",
    "RubricBasedCriterion",
    "SafetyCriterion",
    "SessionInput",
    "SimulationResult",
    "StepType",
    "SyncCriterion",
    "ToolCall",
    "ToolNameMatchCriterion",
    "TrajectoryCollector",
    # Trajectory Criteria
    "TrajectoryMatchCriterion",
    "TrajectoryStep",
    # Simulators
    "UserSimulator",
    "UserSimulatorConfig",
    "WeightedCriterion",
    "print_report",
]
