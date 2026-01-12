# Agent Evaluation Plan for 10xScale Agentflow

## Executive Summary

This document outlines a comprehensive plan for implementing agent evaluation capabilities in Agentflow, inspired by best practices from Google ADK and LangChain/LangSmith. Agent evaluation goes beyond traditional unit testing by addressing the probabilistic nature of LLM-based agents, focusing on trajectory analysis, tool use validation, and response quality assessment.

---

## 1. Research Summary

### 1.1 Google ADK Approach

Google's Agent Development Kit (ADK) provides a robust evaluation framework with the following key components:

#### Evaluation Types
1. **Test Files (Unit Testing)**
   - Single, simple agent-model interactions
   - Designed for rapid execution during development
   - Contains user content, expected tool trajectory, intermediate responses, and final response

2. **EvalSet Files (Integration Testing)**
   - Multiple, potentially lengthy sessions
   - Ideal for complex multi-turn conversations
   - Well-suited for integration tests

#### Evaluation Criteria (Metrics)
| Criterion | Description | Type |
|-----------|-------------|------|
| `tool_trajectory_avg_score` | Exact match of tool call trajectory (EXACT, IN_ORDER, ANY_ORDER) | Deterministic |
| `response_match_score` | ROUGE-1 similarity to reference response | Deterministic |
| `final_response_match_v2` | LLM-judged semantic match to reference | LLM-as-Judge |
| `rubric_based_final_response_quality_v1` | LLM-judged response quality based on custom rubrics | LLM-as-Judge |
| `rubric_based_tool_use_quality_v1` | LLM-judged tool usage quality based on custom rubrics | LLM-as-Judge |
| `hallucinations_v1` | LLM-judged groundedness against context | LLM-as-Judge |
| `safety_v1` | Safety/harmlessness evaluation | LLM-as-Judge |

#### User Simulation
- Dynamic user prompt generation using AI models
- Conversation scenarios with starting prompts and conversation plans
- Useful when fixed prompts aren't practical

### 1.2 LangChain/LangSmith Approach

LangSmith provides three main evaluation types:

#### 1. Final Response Evaluation
- Evaluates the agent's final response against expected output
- Uses LLM-as-judge with custom grading instructions
- Supports semantic equivalence checking

#### 2. Trajectory Evaluation
- Compares actual sequence of steps against expected sequence
- Calculates partial credit for correct steps
- Useful for understanding agent decision-making process

#### 3. Single Step Evaluation
- Evaluates specific components in isolation (e.g., intent classifier, router)
- Enables targeted debugging and iteration
- Direct node/component testing

---

## 2. Evaluation Architecture for Agentflow

### 2.1 Core Components

```
agentflow/
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py           # Main AgentEvaluator class
│   ├── eval_set.py            # EvalSet and EvalCase models
│   ├── eval_config.py         # Evaluation configuration
│   ├── criteria/
│   │   ├── __init__.py
│   │   ├── base.py            # Base criterion interface
│   │   ├── trajectory.py      # Trajectory matching criteria
│   │   ├── response.py        # Response matching criteria
│   │   ├── llm_judge.py       # LLM-as-judge criteria
│   │   └── custom.py          # Custom rubric-based criteria
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── trajectory_collector.py  # Captures execution trajectory
│   │   └── event_collector.py       # Collects all events during execution
│   ├── reporters/
│   │   ├── __init__.py
│   │   ├── console.py         # Console output reporter
│   │   ├── json.py            # JSON file reporter
│   │   └── html.py            # HTML report generator
│   └── simulators/
│       ├── __init__.py
│       └── user_simulator.py  # AI-powered user simulation
```

### 2.2 Data Models

#### EvalCase (Single Test Case)
```python
class EvalCase(BaseModel):
    """A single evaluation case representing one test scenario."""
    eval_id: str
    conversation: list[Invocation]
    session_input: SessionInput
    metadata: dict[str, Any] = {}

class Invocation(BaseModel):
    """A single turn in the conversation."""
    invocation_id: str
    user_content: Message
    expected_tool_trajectory: list[ToolCall] = []
    expected_intermediate_responses: list[Message] = []
    expected_final_response: Message | None = None
```

#### EvalSet (Collection of Test Cases)
```python
class EvalSet(BaseModel):
    """A collection of evaluation cases."""
    eval_set_id: str
    name: str
    description: str = ""
    eval_cases: list[EvalCase]
```

#### EvalConfig (Evaluation Settings)
```python
class EvalConfig(BaseModel):
    """Configuration for evaluation criteria and thresholds."""
    criteria: dict[str, CriterionConfig]
    user_simulator_config: UserSimulatorConfig | None = None
    
class CriterionConfig(BaseModel):
    threshold: float = 0.8
    match_type: str = "EXACT"  # EXACT, IN_ORDER, ANY_ORDER
    judge_model: str = "gpt-4o-mini"
    rubrics: list[Rubric] = []
```

### 2.3 Trajectory Collection

Leverage existing `EventModel` infrastructure to capture execution trajectory:

```python
class TrajectoryCollector:
    """Collects execution trajectory from graph events."""
    
    def __init__(self):
        self.trajectory: list[TrajectoryStep] = []
        self.tool_calls: list[ToolCall] = []
        self.node_visits: list[str] = []
        self.messages: list[Message] = []
    
    async def on_event(self, event: EventModel) -> None:
        """Process incoming events and build trajectory."""
        if event.event == Event.NODE_EXECUTION:
            self.node_visits.append(event.node_name)
            self.trajectory.append(TrajectoryStep(
                step_type="node",
                name=event.node_name,
                timestamp=event.timestamp
            ))
        elif event.event == Event.TOOL_EXECUTION:
            if event.event_type == EventType.START:
                tool_call = ToolCall(
                    name=event.data.get("function_name"),
                    args=event.data.get("args", {}),
                    call_id=event.data.get("tool_call_id")
                )
                self.tool_calls.append(tool_call)
                self.trajectory.append(TrajectoryStep(
                    step_type="tool",
                    name=tool_call.name,
                    args=tool_call.args,
                    timestamp=event.timestamp
                ))
```

---

## 3. Evaluation Criteria Implementation

### 3.1 Trajectory Matching

```python
class TrajectoryMatchCriterion(BaseCriterion):
    """Compare actual vs expected tool call trajectories."""
    
    name = "tool_trajectory_avg_score"
    
    def __init__(self, match_type: str = "EXACT", threshold: float = 1.0):
        self.match_type = match_type  # EXACT, IN_ORDER, ANY_ORDER
        self.threshold = threshold
    
    def evaluate(
        self, 
        actual: list[ToolCall], 
        expected: list[ToolCall]
    ) -> EvalResult:
        if self.match_type == "EXACT":
            score = self._exact_match(actual, expected)
        elif self.match_type == "IN_ORDER":
            score = self._in_order_match(actual, expected)
        else:  # ANY_ORDER
            score = self._any_order_match(actual, expected)
        
        return EvalResult(
            criterion=self.name,
            score=score,
            passed=score >= self.threshold,
            details={
                "actual_trajectory": [t.model_dump() for t in actual],
                "expected_trajectory": [t.model_dump() for t in expected],
                "match_type": self.match_type
            }
        )
    
    def _exact_match(self, actual: list, expected: list) -> float:
        """Require perfect match - same tools, args, and order."""
        if len(actual) != len(expected):
            return 0.0
        matches = sum(1 for a, e in zip(actual, expected) if self._tools_match(a, e))
        return matches / len(expected) if expected else 1.0
    
    def _in_order_match(self, actual: list, expected: list) -> float:
        """Check if expected tools appear in order, allowing extras."""
        if not expected:
            return 1.0
        i = j = 0
        while i < len(expected) and j < len(actual):
            if self._tools_match(expected[i], actual[j]):
                i += 1
            j += 1
        return i / len(expected)
    
    def _any_order_match(self, actual: list, expected: list) -> float:
        """Check if expected tools appear in any order."""
        if not expected:
            return 1.0
        matched = 0
        remaining = list(actual)
        for exp in expected:
            for idx, act in enumerate(remaining):
                if self._tools_match(exp, act):
                    matched += 1
                    remaining.pop(idx)
                    break
        return matched / len(expected)
```

### 3.2 Response Matching

```python
class ResponseMatchCriterion(BaseCriterion):
    """Compare response similarity using ROUGE-1."""
    
    name = "response_match_score"
    
    def evaluate(self, actual: str, expected: str) -> EvalResult:
        # Calculate ROUGE-1 score
        actual_tokens = set(actual.lower().split())
        expected_tokens = set(expected.lower().split())
        
        if not expected_tokens:
            return EvalResult(criterion=self.name, score=1.0, passed=True)
        
        overlap = actual_tokens & expected_tokens
        precision = len(overlap) / len(actual_tokens) if actual_tokens else 0
        recall = len(overlap) / len(expected_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return EvalResult(
            criterion=self.name,
            score=f1,
            passed=f1 >= self.threshold,
            details={"precision": precision, "recall": recall, "f1": f1}
        )
```

### 3.3 LLM-as-Judge

```python
class LLMJudgeCriterion(BaseCriterion):
    """Use an LLM to judge response quality."""
    
    name = "final_response_match_v2"
    
    def __init__(
        self, 
        judge_model: str = "gpt-4o-mini",
        num_samples: int = 5,
        threshold: float = 0.8
    ):
        self.judge_model = judge_model
        self.num_samples = num_samples
        self.threshold = threshold
    
    async def evaluate(
        self,
        question: str,
        actual_response: str,
        expected_response: str
    ) -> EvalResult:
        prompt = f"""You are a teacher grading a quiz.

QUESTION: {question}
GROUND TRUTH RESPONSE: {expected_response}
STUDENT RESPONSE: {actual_response}

Grade criteria:
1. Grade based ONLY on factual accuracy relative to the ground truth.
2. Ensure no conflicting statements in student response.
3. Additional correct information is acceptable.

Return JSON: {{"reasoning": "...", "is_correct": true/false}}"""

        # Sample multiple times and use majority vote
        votes = []
        for _ in range(self.num_samples):
            result = await self._call_judge(prompt)
            votes.append(result["is_correct"])
        
        score = sum(votes) / len(votes)
        return EvalResult(
            criterion=self.name,
            score=score,
            passed=score >= self.threshold,
            details={"votes": votes, "reasoning": result.get("reasoning")}
        )
```

### 3.4 Rubric-Based Evaluation

```python
class RubricCriterion(BaseCriterion):
    """Evaluate against custom rubrics using LLM judge."""
    
    name = "rubric_based_quality"
    
    def __init__(self, rubrics: list[Rubric], judge_model: str = "gpt-4o-mini"):
        self.rubrics = rubrics
        self.judge_model = judge_model
    
    async def evaluate(
        self,
        conversation: list[Message],
        tool_calls: list[ToolCall],
        final_response: str
    ) -> EvalResult:
        rubric_scores = {}
        
        for rubric in self.rubrics:
            prompt = f"""Evaluate the following based on this criterion:

CRITERION: {rubric.content}

CONVERSATION: {self._format_conversation(conversation)}
TOOL CALLS: {self._format_tools(tool_calls)}
FINAL RESPONSE: {final_response}

Does the response satisfy this criterion? Return JSON: {{"verdict": "yes/no", "reasoning": "..."}}"""

            result = await self._call_judge(prompt)
            rubric_scores[rubric.id] = 1.0 if result["verdict"] == "yes" else 0.0
        
        avg_score = sum(rubric_scores.values()) / len(rubric_scores)
        return EvalResult(
            criterion=self.name,
            score=avg_score,
            passed=avg_score >= self.threshold,
            details={"rubric_scores": rubric_scores}
        )
```

---

## 4. Main Evaluator Interface

```python
class AgentEvaluator:
    """Main class for running agent evaluations."""
    
    def __init__(
        self,
        graph: CompiledGraph,
        config: EvalConfig | None = None
    ):
        self.graph = graph
        self.config = config or EvalConfig.default()
        self.criteria = self._build_criteria()
    
    async def evaluate(
        self,
        eval_set: EvalSet | str,  # EvalSet object or path to JSON file
        parallel: bool = False,
        max_concurrency: int = 4
    ) -> EvalReport:
        """Run evaluation on an eval set."""
        if isinstance(eval_set, str):
            eval_set = self._load_eval_set(eval_set)
        
        results = []
        for eval_case in eval_set.eval_cases:
            result = await self._evaluate_case(eval_case)
            results.append(result)
        
        return EvalReport(
            eval_set_id=eval_set.eval_set_id,
            results=results,
            summary=self._compute_summary(results)
        )
    
    async def _evaluate_case(self, case: EvalCase) -> EvalCaseResult:
        """Evaluate a single test case."""
        collector = TrajectoryCollector()
        
        # Run the graph with trajectory collection
        state = AgentState()
        for invocation in case.conversation:
            state.context.append(invocation.user_content)
            
            # Execute graph with event collection
            result = await self.graph.invoke(
                state,
                config={"callbacks": [collector.on_event]}
            )
            
            # Collect actual response
            actual_response = self._extract_response(result)
        
        # Evaluate against all criteria
        criterion_results = []
        for criterion in self.criteria:
            cr_result = await criterion.evaluate(
                actual=collector,
                expected=case
            )
            criterion_results.append(cr_result)
        
        return EvalCaseResult(
            eval_id=case.eval_id,
            passed=all(r.passed for r in criterion_results),
            criterion_results=criterion_results,
            trajectory=collector.trajectory,
            actual_response=actual_response
        )
    
    @classmethod
    async def evaluate_file(
        cls,
        agent_module: str,
        eval_file: str,
        config_file: str | None = None
    ) -> EvalReport:
        """Convenience method to evaluate from file paths."""
        graph = cls._load_graph(agent_module)
        config = cls._load_config(config_file) if config_file else None
        evaluator = cls(graph, config)
        return await evaluator.evaluate(eval_file)
```

---

## 5. Integration with pytest

```python
# tests/evaluation/test_weather_agent.py

import pytest
from agentflow.evaluation import AgentEvaluator

@pytest.mark.asyncio
async def test_weather_agent_basic_trajectory():
    """Test the weather agent follows expected tool trajectory."""
    await AgentEvaluator.evaluate_file(
        agent_module="examples.react.react_weather_agent",
        eval_file="tests/fixtures/weather_agent.test.json"
    )

@pytest.mark.asyncio
async def test_weather_agent_with_custom_criteria():
    """Test with custom evaluation criteria."""
    from agentflow.evaluation import EvalConfig, TrajectoryMatchCriterion
    
    config = EvalConfig(
        criteria={
            "tool_trajectory_avg_score": {
                "threshold": 0.8,
                "match_type": "IN_ORDER"
            },
            "response_match_score": {
                "threshold": 0.7
            }
        }
    )
    
    await AgentEvaluator.evaluate_file(
        agent_module="examples.react.react_weather_agent",
        eval_file="tests/fixtures/weather_agent.test.json",
        config=config
    )
```

---

## 6. CLI Implementation (Developer Guide)

### 6.1 CLI Architecture

```
agentflow/
├── cli/
│   ├── __init__.py
│   ├── main.py              # Entry point, Click app
│   ├── eval_commands.py     # Evaluation commands
│   ├── evalset_commands.py  # EvalSet management
│   ├── simulate_commands.py # User simulation commands
│   ├── utils.py             # CLI utilities
│   └── formatters.py        # Output formatting
```

### 6.2 Main CLI Entry Point

```python
# agentflow/cli/main.py

import click
import asyncio
from pathlib import Path
from typing import Optional

@click.group()
@click.version_option()
def cli():
    """Agentflow - AI Agent Development Framework"""
    pass

@cli.group()
def eval():
    """Agent evaluation commands"""
    pass

@cli.group(name="eval-set")
def eval_set():
    """Manage evaluation sets"""
    pass

@cli.group()
def simulate():
    """User simulation commands"""
    pass

# Register subcommands
from .eval_commands import run_eval, watch_eval, compare_evals
from .evalset_commands import (
    create_evalset, add_case, remove_case, 
    list_cases, validate_evalset, merge_evalsets
)
from .simulate_commands import run_simulation, batch_simulate

eval.add_command(run_eval, "run")
eval.add_command(watch_eval, "watch")
eval.add_command(compare_evals, "compare")

eval_set.add_command(create_evalset, "create")
eval_set.add_command(add_case, "add")
eval_set.add_command(remove_case, "remove")
eval_set.add_command(list_cases, "list")
eval_set.add_command(validate_evalset, "validate")
eval_set.add_command(merge_evalsets, "merge")

simulate.add_command(run_simulation, "run")
simulate.add_command(batch_simulate, "batch")

if __name__ == "__main__":
    cli()
```

### 6.3 Evaluation Commands Implementation

```python
# agentflow/cli/eval_commands.py

import click
import asyncio
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from agentflow.evaluation import AgentEvaluator, EvalConfig, EvalSet
from agentflow.evaluation.reporters import ConsoleReporter, JSONReporter, JUnitXMLReporter

console = Console()

@click.command()
@click.argument("agent_path", type=click.Path(exists=True))
@click.argument("evalset_file", type=click.Path(exists=True))
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to evaluation config JSON file"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path for results (JSON)"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "junit", "html", "console"]),
    multiple=True,
    default=["console"],
    help="Output format(s)"
)
@click.option(
    "--parallel/--sequential",
    default=False,
    help="Run test cases in parallel"
)
@click.option(
    "--max-concurrency",
    type=int,
    default=4,
    help="Max parallel evaluations"
)
@click.option(
    "--filter",
    help="Filter test cases by tag or pattern"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output"
)
@click.option(
    "--fail-fast",
    is_flag=True,
    help="Stop on first failure"
)
@click.option(
    "--min-pass-rate",
    type=float,
    default=1.0,
    help="Minimum pass rate to succeed (0.0-1.0)"
)
@click.option(
    "--save-failures",
    type=click.Path(),
    help="Save failed cases to file for debugging"
)
def run_eval(
    agent_path: str,
    evalset_file: str,
    config: Optional[str],
    output: Optional[str],
    format: tuple[str],
    parallel: bool,
    max_concurrency: int,
    filter: Optional[str],
    verbose: bool,
    fail_fast: bool,
    min_pass_rate: float,
    save_failures: Optional[str]
):
    """Run evaluation on an agent.
    
    AGENT_PATH: Path to agent module or directory
    EVALSET_FILE: Path to evaluation set JSON file
    
    Examples:
        agentflow eval run examples/weather tests/weather.evalset.json
        agentflow eval run my_agent/ tests/eval.json -c config.json -o results.json
        agentflow eval run agent/ tests/ --format json --format junit --parallel
    """
    asyncio.run(_run_eval(
        agent_path, evalset_file, config, output, format,
        parallel, max_concurrency, filter, verbose,
        fail_fast, min_pass_rate, save_failures
    ))

async def _run_eval(
    agent_path: str,
    evalset_file: str,
    config: Optional[str],
    output: Optional[str],
    formats: tuple[str],
    parallel: bool,
    max_concurrency: int,
    filter_pattern: Optional[str],
    verbose: bool,
    fail_fast: bool,
    min_pass_rate: float,
    save_failures: Optional[str]
):
    """Internal async implementation of run_eval."""
    try:
        # Load agent graph
        console.print(f"[blue]Loading agent from {agent_path}...[/blue]")
        graph = await _load_agent_graph(agent_path)
        
        # Load eval config
        if config:
            eval_config = EvalConfig.load(config)
        else:
            eval_config = EvalConfig.default()
        
        # Load eval set
        console.print(f"[blue]Loading eval set from {evalset_file}...[/blue]")
        eval_set = EvalSet.load(evalset_file)
        
        # Apply filter if provided
        if filter_pattern:
            original_count = len(eval_set.eval_cases)
            eval_set = _filter_eval_set(eval_set, filter_pattern)
            console.print(
                f"[yellow]Filtered to {len(eval_set.eval_cases)}/{original_count} cases[/yellow]"
            )
        
        # Create evaluator
        evaluator = AgentEvaluator(graph, eval_config)
        
        # Run evaluation with progress tracking
        console.print(f"[green]Running evaluation...[/green]")
        
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Evaluating...", 
                total=len(eval_set.eval_cases)
            )
            
            report = await evaluator.evaluate(
                eval_set,
                parallel=parallel,
                max_concurrency=max_concurrency,
                fail_fast=fail_fast,
                on_case_complete=lambda _: progress.advance(task)
            )
        
        # Output results
        _output_results(report, formats, output, verbose)
        
        # Save failures if requested
        if save_failures and report.summary.failed_cases > 0:
            _save_failed_cases(report, save_failures)
        
        # Check pass rate
        if report.summary.pass_rate < min_pass_rate:
            console.print(
                f"[red]✗ Pass rate {report.summary.pass_rate:.1%} "
                f"below minimum {min_pass_rate:.1%}[/red]"
            )
            raise SystemExit(1)
        
        console.print(f"[green]✓ Evaluation completed successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise SystemExit(1)

@click.command()
@click.argument("agent_path", type=click.Path(exists=True))
@click.argument("evalset_file", type=click.Path(exists=True))
@click.option(
    "--interval", "-i",
    type=int,
    default=5,
    help="Check interval in seconds"
)
@click.option(
    "--notify",
    is_flag=True,
    help="Send desktop notifications on changes"
)
def watch_eval(agent_path: str, evalset_file: str, interval: int, notify: bool):
    """Watch for changes and re-run evaluations.
    
    Useful during development to get immediate feedback.
    """
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    
    class EvalHandler(FileSystemEventHandler):
        def __init__(self):
            self.last_run = 0
            
        def on_modified(self, event):
            if event.src_path.endswith(('.py', '.json')):
                current = time.time()
                if current - self.last_run > interval:
                    console.print(f"[yellow]Change detected: {event.src_path}[/yellow]")
                    asyncio.run(_run_eval_quick(agent_path, evalset_file))
                    self.last_run = current
    
    observer = Observer()
    observer.schedule(EvalHandler(), agent_path, recursive=True)
    observer.start()
    
    console.print(f"[green]Watching {agent_path} for changes...[/green]")
    try:
        while True:
            time.sleep(interval)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

@click.command()
@click.argument("report_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--baseline",
    type=click.Path(exists=True),
    help="Baseline report to compare against"
)
@click.option(
    "--threshold",
    type=float,
    default=0.05,
    help="Regression threshold (default: 0.05)"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Save comparison to file"
)
def compare_evals(
    report_files: tuple[str],
    baseline: Optional[str],
    threshold: float,
    output: Optional[str]
):
    """Compare evaluation reports to detect regressions.
    
    Examples:
        agentflow eval compare results/report1.json results/report2.json
        agentflow eval compare results/latest.json --baseline results/baseline.json
    """
    from agentflow.evaluation import EvalReport
    from agentflow.cli.formatters import format_comparison
    
    if baseline:
        baseline_report = EvalReport.load(baseline)
        reports = [EvalReport.load(f) for f in report_files]
        
        for report in reports:
            comparison = _compare_reports(baseline_report, report, threshold)
            console.print(format_comparison(comparison))
    else:
        reports = [EvalReport.load(f) for f in report_files]
        comparison = _compare_multiple_reports(reports, threshold)
        console.print(format_comparison(comparison))
    
    if output:
        # Save comparison result
        pass

# Helper functions

async def _load_agent_graph(agent_path: str):
    """Load and compile agent graph from path."""
    from importlib import import_module
    from pathlib import Path
    
    path = Path(agent_path)
    
    if path.is_file():
        # Load from Python file
        module_path = str(path.with_suffix('').as_posix()).replace('/', '.')
        module = import_module(module_path)
        
        # Look for create_graph or similar function
        for attr_name in ['create_graph', 'create_agent', 'main', 'graph']:
            if hasattr(module, attr_name):
                graph_or_func = getattr(module, attr_name)
                if callable(graph_or_func):
                    result = graph_or_func()
                    if asyncio.iscoroutine(result):
                        result = await result
                    return result
                return graph_or_func
        
        raise ValueError(f"No graph creation function found in {agent_path}")
    
    elif path.is_dir():
        # Look for main.py or __init__.py
        for filename in ['main.py', '__init__.py', 'agent.py']:
            file_path = path / filename
            if file_path.exists():
                return await _load_agent_graph(str(file_path))
        
        raise ValueError(f"No agent module found in {agent_path}")
    
    else:
        raise ValueError(f"Invalid agent path: {agent_path}")

def _filter_eval_set(eval_set: EvalSet, pattern: str) -> EvalSet:
    """Filter eval set by tag or case ID pattern."""
    import fnmatch
    
    filtered_cases = [
        case for case in eval_set.eval_cases
        if (
            fnmatch.fnmatch(case.eval_id, pattern) or
            fnmatch.fnmatch(case.name or "", pattern) or
            any(fnmatch.fnmatch(tag, pattern) for tag in case.tags)
        )
    ]
    
    return EvalSet(
        eval_set_id=eval_set.eval_set_id,
        name=eval_set.name,
        description=eval_set.description,
        eval_cases=filtered_cases,
        metadata=eval_set.metadata
    )

def _output_results(report, formats: tuple[str], output: Optional[str], verbose: bool):
    """Output results in requested formats."""
    from agentflow.evaluation.reporters import (
        ConsoleReporter, JSONReporter, JUnitXMLReporter, HTMLReporter
    )
    
    for fmt in formats:
        if fmt == "console":
            reporter = ConsoleReporter(verbose=verbose)
            reporter.report(report)
        
        elif fmt == "json":
            reporter = JSONReporter()
            path = output or "eval_report.json"
            reporter.save(report, path)
            console.print(f"[green]JSON report saved to {path}[/green]")
        
        elif fmt == "junit":
            reporter = JUnitXMLReporter()
            path = output or "junit.xml"
            reporter.save(report, path)
            console.print(f"[green]JUnit XML saved to {path}[/green]")
        
        elif fmt == "html":
            reporter = HTMLReporter()
            path = output or "eval_report.html"
            reporter.save(report, path)
            console.print(f"[green]HTML report saved to {path}[/green]")

def _save_failed_cases(report, filepath: str):
    """Save failed cases to a new eval set for debugging."""
    from agentflow.evaluation import EvalSet
    
    failed_cases = [
        result.eval_case for result in report.results
        if not result.passed
    ]
    
    failed_set = EvalSet(
        eval_set_id=f"{report.eval_set_id}_failures",
        name=f"Failed Cases from {report.eval_set_name}",
        description="Auto-generated set of failed test cases",
        eval_cases=failed_cases,
        metadata={
            "original_eval_set_id": report.eval_set_id,
            "created_from_report": report.report_id,
        }
    )
    
    failed_set.save(filepath)
    console.print(f"[yellow]Failed cases saved to {filepath}[/yellow]")
```

### 6.4 EvalSet Management Commands

```python
# agentflow/cli/evalset_commands.py

import click
from rich.console import Console
from rich.table import Table
from agentflow.evaluation import EvalSet, EvalCase

console = Console()

@click.command()
@click.argument("name")
@click.option(
    "--description", "-d",
    help="Description of the eval set"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output file path"
)
def create_evalset(name: str, description: str, output: str):
    """Create a new empty evaluation set.
    
    Example:
        agentflow eval-set create my_tests -d "My test suite" -o tests/my_tests.evalset.json
    """
    eval_set = EvalSet(
        eval_set_id=name,
        name=name,
        description=description or "",
        eval_cases=[]
    )
    
    path = output or f"{name}.evalset.json"
    eval_set.save(path)
    
    console.print(f"[green]✓ Created eval set: {path}[/green]")

@click.command()
@click.argument("evalset_file", type=click.Path(exists=True))
@click.option(
    "--from-session",
    type=click.Path(exists=True),
    help="Create case from recorded session"
)
@click.option(
    "--from-template",
    type=click.Choice(["single-turn", "multi-turn", "tool-test"]),
    help="Create case from template"
)
@click.option(
    "--interactive", "-i",
    is_flag=True,
    help="Interactive case creation"
)
def add_case(evalset_file: str, from_session: str, from_template: str, interactive: bool):
    """Add a test case to an eval set.
    
    Examples:
        agentflow eval-set add tests/my_tests.evalset.json --from-session session.json
        agentflow eval-set add tests/my_tests.evalset.json --from-template single-turn
        agentflow eval-set add tests/my_tests.evalset.json --interactive
    """
    eval_set = EvalSet.load(evalset_file)
    
    if from_session:
        case = _create_case_from_session(from_session)
    elif from_template:
        case = _create_case_from_template(from_template)
    elif interactive:
        case = _create_case_interactive()
    else:
        console.print("[red]Must specify one of: --from-session, --from-template, --interactive[/red]")
        raise SystemExit(1)
    
    eval_set.eval_cases.append(case)
    eval_set.save(evalset_file)
    
    console.print(f"[green]✓ Added case '{case.eval_id}' to {evalset_file}[/green]")

@click.command()
@click.argument("evalset_file", type=click.Path(exists=True))
@click.argument("case_id")
def remove_case(evalset_file: str, case_id: str):
    """Remove a test case from an eval set."""
    eval_set = EvalSet.load(evalset_file)
    
    original_count = len(eval_set.eval_cases)
    eval_set.eval_cases = [
        case for case in eval_set.eval_cases
        if case.eval_id != case_id
    ]
    
    if len(eval_set.eval_cases) == original_count:
        console.print(f"[yellow]Case '{case_id}' not found[/yellow]")
        raise SystemExit(1)
    
    eval_set.save(evalset_file)
    console.print(f"[green]✓ Removed case '{case_id}'[/green]")

@click.command()
@click.argument("evalset_file", type=click.Path(exists=True))
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed information"
)
@click.option(
    "--filter",
    help="Filter by tag or pattern"
)
def list_cases(evalset_file: str, verbose: bool, filter: str):
    """List all test cases in an eval set.
    
    Example:
        agentflow eval-set list tests/my_tests.evalset.json
        agentflow eval-set list tests/my_tests.evalset.json --filter "weather*"
    """
    eval_set = EvalSet.load(evalset_file)
    
    table = Table(title=f"Eval Set: {eval_set.name}")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Turns", justify="right")
    table.add_column("Tags")
    
    cases = eval_set.eval_cases
    if filter:
        import fnmatch
        cases = [c for c in cases if fnmatch.fnmatch(c.eval_id, filter)]
    
    for case in cases:
        table.add_row(
            case.eval_id,
            case.name or "-",
            str(len(case.conversation)),
            ", ".join(case.tags) if case.tags else "-"
        )
    
    console.print(table)
    console.print(f"\n[blue]Total: {len(cases)} cases[/blue]")
    
    if verbose:
        # Show more details
        for case in cases:
            console.print(f"\n[bold]{case.eval_id}[/bold]")
            console.print(f"  Name: {case.name or 'N/A'}")
            console.print(f"  Turns: {len(case.conversation)}")
            console.print(f"  Tags: {', '.join(case.tags) if case.tags else 'None'}")
            
            if case.metadata:
                console.print(f"  Metadata: {case.metadata}")

@click.command()
@click.argument("evalset_file", type=click.Path(exists=True))
@click.option(
    "--fix",
    is_flag=True,
    help="Attempt to fix validation errors"
)
def validate_evalset(evalset_file: str, fix: bool):
    """Validate an eval set file.
    
    Checks for:
    - Valid JSON structure
    - Required fields present
    - Valid data types
    - Consistent IDs
    - Valid tool trajectories
    """
    try:
        eval_set = EvalSet.load(evalset_file)
        errors = _validate_eval_set_structure(eval_set)
        
        if not errors:
            console.print(f"[green]✓ {evalset_file} is valid[/green]")
            return
        
        console.print(f"[red]Found {len(errors)} validation errors:[/red]")
        for error in errors:
            console.print(f"  • {error}")
        
        if fix:
            console.print("\n[yellow]Attempting to fix errors...[/yellow]")
            fixed_set = _fix_eval_set(eval_set, errors)
            fixed_set.save(evalset_file)
            console.print(f"[green]✓ Fixed and saved[/green]")
        else:
            console.print("\n[yellow]Run with --fix to attempt auto-repair[/yellow]")
            raise SystemExit(1)
            
    except Exception as e:
        console.print(f"[red]Error loading eval set: {e}[/red]")
        raise SystemExit(1)

@click.command()
@click.argument("evalset_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(),
    help="Output file for merged eval set"
)
@click.option(
    "--deduplicate",
    is_flag=True,
    help="Remove duplicate cases"
)
def merge_evalsets(evalset_files: tuple[str], output: str, deduplicate: bool):
    """Merge multiple eval sets into one.
    
    Example:
        agentflow eval-set merge tests/*.evalset.json -o tests/all_tests.evalset.json
    """
    all_cases = []
    seen_ids = set()
    
    for filepath in evalset_files:
        eval_set = EvalSet.load(filepath)
        for case in eval_set.eval_cases:
            if deduplicate and case.eval_id in seen_ids:
                continue
            all_cases.append(case)
            seen_ids.add(case.eval_id)
    
    merged = EvalSet(
        eval_set_id="merged",
        name="Merged Eval Set",
        description=f"Merged from {len(evalset_files)} eval sets",
        eval_cases=all_cases
    )
    
    merged.save(output)
    console.print(f"[green]✓ Merged {len(all_cases)} cases to {output}[/green]")

# Helper functions

def _create_case_from_session(session_file: str) -> EvalCase:
    """Create eval case from recorded session."""
    # Load session data and convert to EvalCase
    pass

def _create_case_from_template(template: str) -> EvalCase:
    """Create eval case from template."""
    pass

def _create_case_interactive() -> EvalCase:
    """Interactively create eval case."""
    from InquirerPy import prompt
    
    questions = [
        {
            "type": "input",
            "name": "eval_id",
            "message": "Case ID:",
        },
        {
            "type": "input",
            "name": "name",
            "message": "Case name:",
        },
        # ... more questions
    ]
    
    answers = prompt(questions)
    # Create EvalCase from answers
    pass

def _validate_eval_set_structure(eval_set: EvalSet) -> list[str]:
    """Validate eval set structure."""
    errors = []
    
    # Check for duplicate IDs
    ids = [case.eval_id for case in eval_set.eval_cases]
    duplicates = [id for id in ids if ids.count(id) > 1]
    if duplicates:
        errors.append(f"Duplicate case IDs: {duplicates}")
    
    # Validate each case
    for case in eval_set.eval_cases:
        if not case.conversation:
            errors.append(f"Case '{case.eval_id}' has no conversation turns")
        
        for invocation in case.conversation:
            if not invocation.user_content:
                errors.append(f"Case '{case.eval_id}' has invocation without user content")
    
    return errors

def _fix_eval_set(eval_set: EvalSet, errors: list[str]) -> EvalSet:
    """Attempt to fix eval set errors."""
    # Implement auto-fix logic
    pass
```

### 6.5 User Simulation Commands

```python
# agentflow/cli/simulate_commands.py

import click
import asyncio
from rich.console import Console
from agentflow.evaluation import UserSimulator, ConversationScenario

console = Console()

@click.command()
@click.argument("agent_path", type=click.Path(exists=True))
@click.option(
    "--prompt",
    required=True,
    help="Starting user prompt"
)
@click.option(
    "--plan",
    help="Conversation plan (multi-line text)"
)
@click.option(
    "--goals",
    help="Comma-separated list of goals"
)
@click.option(
    "--max-turns",
    type=int,
    default=10,
    help="Maximum conversation turns"
)
@click.option(
    "--model",
    default="gpt-4o-mini",
    help="Model for user simulation"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Save simulation result to file"
)
@click.option(
    "--save-as-eval",
    type=click.Path(),
    help="Convert successful simulation to eval case"
)
def run_simulation(
    agent_path: str,
    prompt: str,
    plan: str,
    goals: str,
    max_turns: int,
    model: str,
    output: str,
    save_as_eval: str
):
    """Run user simulation against an agent.
    
    Example:
        agentflow simulate run agent/ --prompt "What's the weather?" --goals "temperature"
    """
    asyncio.run(_run_simulation(
        agent_path, prompt, plan, goals,
        max_turns, model, output, save_as_eval
    ))

async def _run_simulation(
    agent_path: str,
    prompt: str,
    plan: str,
    goals_str: str,
    max_turns: int,
    model: str,
    output: str,
    save_as_eval: str
):
    """Internal async implementation."""
    from agentflow.cli.eval_commands import _load_agent_graph
    
    # Load agent
    console.print(f"[blue]Loading agent from {agent_path}...[/blue]")
    graph = await _load_agent_graph(agent_path)
    
    # Create scenario
    goals = goals_str.split(",") if goals_str else []
    scenario = ConversationScenario(
        scenario_id="cli_simulation",
        description="User simulation from CLI",
        starting_prompt=prompt,
        conversation_plan=plan or "",
        goals=goals,
        max_turns=max_turns
    )
    
    # Run simulation
    simulator = UserSimulator(model=model)
    
    console.print(f"[green]Running simulation...[/green]")
    result = await simulator.run(graph, scenario)
    
    # Display results
    _display_simulation_result(result)
    
    # Save if requested
    if output:
        import json
        with open(output, "w") as f:
            json.dump(result.model_dump(), f, indent=2)
        console.print(f"[green]Saved to {output}[/green]")
    
    if save_as_eval and result.completed:
        _save_as_eval_case(result, save_as_eval)

def _display_simulation_result(result):
    """Display simulation result in console."""
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    # Status
    status = "✓ Completed" if result.completed else "✗ Incomplete"
    color = "green" if result.completed else "red"
    
    console.print(f"\n[{color}]{status}[/{color}]")
    console.print(f"Turns: {result.turns}")
    console.print(f"Goals achieved: {len(result.goals_achieved)}/{len(result.goals)}")
    
    # Conversation
    console.print("\n[bold]Conversation:[/bold]")
    for i, msg in enumerate(result.conversation, 1):
        role = msg["role"].upper()
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        
        console.print(Panel(
            content,
            title=f"Turn {i}: {role}",
            border_style="blue" if role == "USER" else "green"
        ))

@click.command()
@click.argument("agent_path", type=click.Path(exists=True))
@click.argument("scenarios_file", type=click.Path(exists=True))
@click.option(
    "--parallel",
    is_flag=True,
    help="Run simulations in parallel"
)
@click.option(
    "--max-concurrency",
    type=int,
    default=4,
    help="Max parallel simulations"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output directory for results"
)
@click.option(
    "--save-successful",
    type=click.Path(),
    help="Save successful simulations as eval set"
)
def batch_simulate(
    agent_path: str,
    scenarios_file: str,
    parallel: bool,
    max_concurrency: int,
    output: str,
    save_successful: str
):
    """Run batch user simulations from scenarios file.
    
    SCENARIOS_FILE should be a JSON file with list of scenarios.
    
    Example:
        agentflow simulate batch agent/ scenarios.json -o results/
    """
    asyncio.run(_batch_simulate(
        agent_path, scenarios_file, parallel,
        max_concurrency, output, save_successful
    ))

async def _batch_simulate(
    agent_path: str,
    scenarios_file: str,
    parallel: bool,
    max_concurrency: int,
    output: str,
    save_successful: str
):
    """Internal async implementation."""
    from agentflow.evaluation import BatchSimulator
    from agentflow.cli.eval_commands import _load_agent_graph
    import json
    
    # Load agent and scenarios
    graph = await _load_agent_graph(agent_path)
    
    with open(scenarios_file) as f:
        scenarios_data = json.load(f)
    
    scenarios = [
        ConversationScenario.model_validate(s)
        for s in scenarios_data
    ]
    
    # Run batch simulation
    console.print(f"[green]Running {len(scenarios)} simulations...[/green]")
    
    simulator = BatchSimulator()
    results = await simulator.run_all(
        graph,
        scenarios,
        parallel=parallel,
        max_concurrency=max_concurrency
    )
    
    # Analyze results
    successful = [r for r in results if r.completed]
    failed = [r for r in results if not r.completed]
    
    console.print(f"\n[green]Successful: {len(successful)}/{len(results)}[/green]")
    if failed:
        console.print(f"[red]Failed: {len(failed)}[/red]")
    
    # Save results
    if output:
        Path(output).mkdir(exist_ok=True, parents=True)
        for result in results:
            filepath = Path(output) / f"{result.scenario_id}.json"
            with open(filepath, "w") as f:
                json.dump(result.model_dump(), f, indent=2)
    
    if save_successful and successful:
        _save_simulations_as_evalset(successful, save_successful)
```

### 6.6 CLI Utilities

```python
# agentflow/cli/utils.py

from pathlib import Path
from typing import Any, Optional
import sys

def ensure_output_dir(filepath: str) -> Path:
    """Ensure output directory exists."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def load_json_file(filepath: str) -> dict[str, Any]:
    """Load JSON file with error handling."""
    import json
    try:
        with open(filepath) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

def save_json_file(data: dict[str, Any], filepath: str, indent: int = 2):
    """Save data to JSON file."""
    import json
    ensure_output_dir(filepath)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=indent)

def find_agent_module(path: str) -> str:
    """Find agent module from path."""
    from pathlib import Path
    
    path_obj = Path(path)
    
    # Check common patterns
    candidates = [
        path_obj / "main.py",
        path_obj / "agent.py",
        path_obj / "__init__.py",
        path_obj if path_obj.is_file() else None
    ]
    
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    
    raise ValueError(f"No agent module found in {path}")

def format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_percentage(value: float) -> str:
    """Format percentage with color."""
    percentage = value * 100
    if percentage >= 95:
        color = "green"
    elif percentage >= 80:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{percentage:.1f}%[/{color}]"
```

### 6.7 Setup Entry Point

```python
# setup.py additions for CLI

entry_points={
    "console_scripts": [
        "agentflow=agentflow.cli.main:cli",
    ],
}
```

---

## 7. Architecture Deep Dive (Developer Focus)

### 7.1 Event-Driven Trajectory Collection

The trajectory collection system integrates deeply with Agentflow's event system:

```python
# agentflow/evaluation/collectors/trajectory_collector.py

from typing import Protocol
from agentflow.core.events import EventModel, Event, EventType

class EventCollector(Protocol):
    """Protocol for event collectors."""
    async def on_event(self, event: EventModel) -> None: ...
    def get_trajectory(self) -> list[TrajectoryStep]: ...

class TrajectoryCollector:
    """
    Collects execution trajectory by subscribing to graph events.
    
    Architecture:
    1. Subscribes to EventModel stream via callback
    2. Filters relevant events (NODE_EXECUTION, TOOL_EXECUTION, etc.)
    3. Builds structured trajectory in real-time
    4. Maintains state for correlation (tool call IDs, timestamps)
    
    Thread Safety: Not thread-safe. Use one collector per evaluation task.
    """
    
    def __init__(self):
        # State tracking
        self.trajectory: list[TrajectoryStep] = []
        self.tool_calls: dict[str, ToolCall] = {}  # call_id -> ToolCall
        self.node_visits: list[str] = []
        self.messages: list[Message] = []
        
        # Timing
        self.start_time: float | None = None
        self.end_time: float | None = None
        
        # Pending state (for correlating START/END events)
        self._pending_tools: dict[str, dict] = {}
        self._pending_nodes: dict[str, dict] = {}
    
    async def on_event(self, event: EventModel) -> None:
        """
        Process incoming events.
        
        Event Flow:
        1. NODE_EXECUTION (START) -> Record node entry
        2. TOOL_EXECUTION (START) -> Create pending tool call
        3. TOOL_EXECUTION (END) -> Complete tool call with result
        4. MESSAGE_GENERATION -> Capture model output
        5. NODE_EXECUTION (END) -> Record node exit
        
        This method MUST be fast as it's called synchronously in the event loop.
        """
        if self.start_time is None:
            self.start_time = event.timestamp
        
        self.end_time = event.timestamp
        
        # Dispatch to specific handlers
        if event.event == Event.NODE_EXECUTION:
            await self._handle_node_event(event)
        elif event.event == Event.TOOL_EXECUTION:
            await self._handle_tool_event(event)
        elif event.event == Event.MESSAGE_GENERATION:
            await self._handle_message_event(event)
        elif event.event == Event.CONDITIONAL_EDGE:
            await self._handle_conditional_event(event)
    
    async def _handle_node_event(self, event: EventModel) -> None:
        """Handle node execution events."""
        if event.event_type == EventType.START:
            self.node_visits.append(event.node_name)
            self._pending_nodes[event.node_name] = {
                "start_time": event.timestamp,
                "metadata": event.data
            }
            
            self.trajectory.append(TrajectoryStep(
                step_type=StepType.NODE,
                name=event.node_name,
                timestamp=event.timestamp,
                metadata={"state": "start"}
            ))
            
        elif event.event_type == EventType.END:
            if event.node_name in self._pending_nodes:
                pending = self._pending_nodes.pop(event.node_name)
                duration = event.timestamp - pending["start_time"]
                
                self.trajectory.append(TrajectoryStep(
                    step_type=StepType.NODE,
                    name=event.node_name,
                    timestamp=event.timestamp,
                    metadata={
                        "state": "end",
                        "duration": duration
                    }
                ))
    
    async def _handle_tool_event(self, event: EventModel) -> None:
        """Handle tool execution events with call ID correlation."""
        if event.event_type == EventType.START:
            call_id = event.data.get("tool_call_id") or event.data.get("id")
            
            tool_call = ToolCall(
                name=event.data.get("function_name") or event.data.get("name"),
                args=event.data.get("args", {}),
                call_id=call_id
            )
            
            self._pending_tools[call_id] = {
                "tool_call": tool_call,
                "start_time": event.timestamp
            }
            
            self.trajectory.append(TrajectoryStep(
                step_type=StepType.TOOL,
                name=tool_call.name,
                args=tool_call.args,
                timestamp=event.timestamp,
                metadata={"call_id": call_id, "state": "start"}
            ))
            
        elif event.event_type == EventType.END:
            call_id = event.data.get("tool_call_id") or event.data.get("id")
            
            if call_id in self._pending_tools:
                pending = self._pending_tools.pop(call_id)
                tool_call = pending["tool_call"]
                tool_call.result = event.data.get("result")
                
                self.tool_calls[call_id] = tool_call
                
                duration = event.timestamp - pending["start_time"]
                
                self.trajectory.append(TrajectoryStep(
                    step_type=StepType.TOOL,
                    name=tool_call.name,
                    args=tool_call.args,
                    timestamp=event.timestamp,
                    metadata={
                        "call_id": call_id,
                        "state": "end",
                        "duration": duration,
                        "result": tool_call.result
                    }
                ))
    
    async def _handle_message_event(self, event: EventModel) -> None:
        """Handle message generation events."""
        message = event.data.get("message")
        if message:
            self.messages.append(message)
            
            self.trajectory.append(TrajectoryStep(
                step_type=StepType.MESSAGE,
                name="message",
                timestamp=event.timestamp,
                metadata={"content": message}
            ))
    
    async def _handle_conditional_event(self, event: EventModel) -> None:
        """Handle conditional edge routing."""
        self.trajectory.append(TrajectoryStep(
            step_type=StepType.CONDITIONAL,
            name=event.data.get("condition", "unknown"),
            timestamp=event.timestamp,
            metadata={
                "from_node": event.data.get("from_node"),
                "to_node": event.data.get("to_node"),
                "condition_result": event.data.get("result")
            }
        ))
    
    def get_tool_calls(self) -> list[ToolCall]:
        """Get completed tool calls in chronological order."""
        return sorted(
            self.tool_calls.values(),
            key=lambda tc: self._get_tool_timestamp(tc.call_id)
        )
    
    def _get_tool_timestamp(self, call_id: str) -> float:
        """Get timestamp for tool call from trajectory."""
        for step in self.trajectory:
            if (step.step_type == StepType.TOOL and 
                step.metadata.get("call_id") == call_id and
                step.metadata.get("state") == "start"):
                return step.timestamp
        return 0.0
```

### 7.2 Criterion Evaluation Pipeline

```python
# agentflow/evaluation/evaluator.py - Internal pipeline

class EvaluationPipeline:
    """
    Orchestrates the evaluation of a single test case through multiple criteria.
    
    Design Decisions:
    1. Sequential execution by default (ensures LLM rate limits respected)
    2. Optional parallel execution for independent criteria
    3. Early termination on critical failures (configurable)
    4. Automatic retry with exponential backoff for transient failures
    """
    
    def __init__(
        self,
        criteria: list[BaseCriterion],
        parallel: bool = False,
        fail_fast: bool = False,
        retry_config: RetryConfig | None = None
    ):
        self.criteria = criteria
        self.parallel = parallel
        self.fail_fast = fail_fast
        self.retry_config = retry_config or RetryConfig.default()
    
    async def evaluate_case(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
        context: EvaluationContext
    ) -> list[CriterionResult]:
        """
        Run all criteria against a test case.
        
        Returns:
            List of criterion results (may be incomplete if fail_fast=True)
        """
        if self.parallel:
            return await self._evaluate_parallel(actual, expected, context)
        else:
            return await self._evaluate_sequential(actual, expected, context)
    
    async def _evaluate_sequential(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
        context: EvaluationContext
    ) -> list[CriterionResult]:
        """Sequential evaluation with retry logic."""
        results = []
        
        for criterion in self.criteria:
            try:
                # Retry wrapper
                result = await self._evaluate_with_retry(
                    criterion, actual, expected, context
                )
                
                results.append(result)
                
                # Early termination check
                if self.fail_fast and not result.passed:
                    context.logger.info(
                        f"Fail-fast triggered on criterion '{criterion.name}'"
                    )
                    break
                    
            except Exception as e:
                # Create error result
                results.append(CriterionResult.error(
                    criterion=criterion.name,
                    error=str(e),
                    threshold=criterion.config.threshold
                ))
                
                if self.fail_fast:
                    break
        
        return results
    
    async def _evaluate_parallel(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
        context: EvaluationContext
    ) -> list[CriterionResult]:
        """Parallel evaluation using asyncio.gather."""
        tasks = [
            self._evaluate_with_retry(criterion, actual, expected, context)
            for criterion in self.criteria
        ]
        
        # Use return_exceptions to handle individual failures
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for criterion, result in zip(self.criteria, results):
            if isinstance(result, Exception):
                final_results.append(CriterionResult.error(
                    criterion=criterion.name,
                    error=str(result),
                    threshold=criterion.config.threshold
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def _evaluate_with_retry(
        self,
        criterion: BaseCriterion,
        actual: TrajectoryCollector,
        expected: EvalCase,
        context: EvaluationContext
    ) -> CriterionResult:
        """Evaluate with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Add timeout
                result = await asyncio.wait_for(
                    criterion.evaluate(actual, expected, context),
                    timeout=self.retry_config.timeout_seconds
                )
                return result
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.retry_config.timeout_seconds}s"
                
            except RateLimitError as e:
                # Special handling for rate limits
                last_error = str(e)
                delay = self.retry_config.backoff_base ** attempt
                await asyncio.sleep(delay)
                
            except Exception as e:
                # Non-retryable errors
                if not self._is_retryable(e):
                    raise
                last_error = str(e)
        
        raise EvaluationError(
            f"Criterion '{criterion.name}' failed after {attempt + 1} attempts: {last_error}"
        )
    
    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Check if error is retryable."""
        retryable_types = (
            TimeoutError,
            ConnectionError,
            # Add LLM provider specific errors
        )
        return isinstance(error, retryable_types)


class EvaluationContext:
    """
    Context object passed through evaluation pipeline.
    
    Provides:
    - Logger for debugging
    - Metrics collector
    - Shared state between criteria
    - Cancel token for early termination
    """
    
    def __init__(self):
        self.logger = logging.getLogger("agentflow.evaluation")
        self.metrics = MetricsCollector()
        self.shared_state: dict[str, Any] = {}
        self.cancel_token = CancelToken()
    
    def set_shared(self, key: str, value: Any) -> None:
        """Set shared state for use by other criteria."""
        self.shared_state[key] = value
    
    def get_shared(self, key: str, default: Any = None) -> Any:
        """Get shared state."""
        return self.shared_state.get(key, default)
```

### 7.3 Integration with Existing Graph System

```python
# Integration patterns with CompiledGraph

class GraphEvaluationAdapter:
    """
    Adapter to integrate evaluation system with existing graph infrastructure.
    
    Responsibilities:
    1. Inject trajectory collector into graph execution
    2. Handle state management across multi-turn conversations
    3. Extract results from graph output
    4. Clean up resources
    """
    
    def __init__(self, graph: CompiledGraph):
        self.graph = graph
        self._original_callbacks = None
    
    async def execute_with_collection(
        self,
        initial_state: AgentState,
        invocations: list[Invocation],
        collector: TrajectoryCollector
    ) -> tuple[AgentState, list[Any]]:
        """
        Execute graph with trajectory collection across multiple invocations.
        
        Args:
            initial_state: Starting agent state
            invocations: List of user messages/invocations
            collector: Trajectory collector instance
        
        Returns:
            Final state and list of responses
        """
        current_state = initial_state
        responses = []
        
        try:
            # Inject collector as callback
            self._inject_collector(collector)
            
            for invocation in invocations:
                # Add user message to state
                current_state = self._add_user_message(
                    current_state,
                    invocation.user_content
                )
                
                # Execute graph
                result = await self.graph.ainvoke(
                    current_state,
                    config={
                        "callbacks": [collector.on_event],
                        "recursion_limit": 100,
                    }
                )
                
                # Extract response
                response = self._extract_response(result)
                responses.append(response)
                
                # Update state for next turn
                current_state = result
            
            return current_state, responses
            
        finally:
            # Clean up
            self._restore_callbacks()
    
    def _inject_collector(self, collector: TrajectoryCollector) -> None:
        """Inject collector into graph callback chain."""
        # Store original callbacks
        self._original_callbacks = getattr(self.graph, "_callbacks", [])
        
        # Add collector to callback chain
        callbacks = list(self._original_callbacks)
        callbacks.append(collector.on_event)
        self.graph._callbacks = callbacks
    
    def _restore_callbacks(self) -> None:
        """Restore original callback configuration."""
        if self._original_callbacks is not None:
            self.graph._callbacks = self._original_callbacks
            self._original_callbacks = None
    
    def _add_user_message(
        self,
        state: AgentState,
        message: Message
    ) -> AgentState:
        """Add user message to agent state."""
        # This depends on your AgentState structure
        new_state = state.copy(deep=True)
        new_state.context.append(message)
        return new_state
    
    def _extract_response(self, result: AgentState) -> Message:
        """Extract agent response from result state."""
        # This depends on your AgentState structure
        if hasattr(result, "response"):
            return result.response
        elif hasattr(result, "messages") and result.messages:
            return result.messages[-1]
        else:
            raise ValueError("Cannot extract response from result state")


# Usage in AgentEvaluator

class AgentEvaluator:
    """Main evaluator class."""
    
    async def _evaluate_case(self, case: EvalCase) -> EvalCaseResult:
        """Evaluate single case using adapter."""
        collector = TrajectoryCollector()
        adapter = GraphEvaluationAdapter(self.graph)
        
        try:
            # Create initial state
            initial_state = self._create_initial_state(case.session_input)
            
            # Execute with collection
            final_state, responses = await adapter.execute_with_collection(
                initial_state,
                case.conversation,
                collector
            )
            
            # Create evaluation context
            context = EvaluationContext()
            context.set_shared("responses", responses)
            context.set_shared("final_state", final_state)
            
            # Run criteria pipeline
            pipeline = EvaluationPipeline(
                self.criteria,
                parallel=self.config.parallel_criteria,
                fail_fast=self.config.fail_fast
            )
            
            criterion_results = await pipeline.evaluate_case(
                collector, case, context
            )
            
            # Build result
            return EvalCaseResult(
                eval_id=case.eval_id,
                name=case.name,
                passed=all(r.passed for r in criterion_results),
                criterion_results=criterion_results,
                trajectory=collector.trajectory,
                actual_responses=responses,
                duration_seconds=collector.end_time - collector.start_time
            )
            
        except Exception as e:
            # Return error result
            return EvalCaseResult.error(
                eval_id=case.eval_id,
                name=case.name,
                error=str(e)
            )
```

### 7.4 Error Handling Strategy

```python
# agentflow/evaluation/errors.py

class EvaluationError(Exception):
    """Base exception for evaluation errors."""
    pass

class GraphExecutionError(EvaluationError):
    """Error during graph execution."""
    def __init__(self, message: str, original_error: Exception):
        super().__init__(message)
        self.original_error = original_error

class CriterionEvaluationError(EvaluationError):
    """Error during criterion evaluation."""
    def __init__(self, criterion_name: str, message: str):
        super().__init__(f"Criterion '{criterion_name}': {message}")
        self.criterion_name = criterion_name

class RateLimitError(EvaluationError):
    """Rate limit exceeded (for LLM-as-judge)."""
    def __init__(self, retry_after: float | None = None):
        super().__init__("Rate limit exceeded")
        self.retry_after = retry_after

class InvalidEvalSetError(EvaluationError):
    """Invalid eval set structure."""
    pass

class TimeoutError(EvaluationError):
    """Evaluation timeout."""
    pass

# Error handling in evaluator

class AgentEvaluator:
    """Evaluator with comprehensive error handling."""
    
    async def evaluate(
        self,
        eval_set: EvalSet | str,
        error_strategy: ErrorStrategy = ErrorStrategy.COLLECT
    ) -> EvalReport:
        """
        Evaluate with configurable error handling.
        
        Args:
            eval_set: Evaluation set
            error_strategy: How to handle errors
                - FAIL_FAST: Stop on first error
                - COLLECT: Collect all errors, continue evaluation
                - SKIP: Skip failed cases, continue with rest
        
        Returns:
            Evaluation report (may include error results)
        """
        results = []
        errors = []
        
        for case in eval_set.eval_cases:
            try:
                result = await self._evaluate_case_safe(case)
                results.append(result)
                
            except Exception as e:
                error_info = ErrorInfo(
                    case_id=case.eval_id,
                    error=e,
                    timestamp=time.time()
                )
                errors.append(error_info)
                
                if error_strategy == ErrorStrategy.FAIL_FAST:
                    raise EvaluationError(
                        f"Evaluation failed at case '{case.eval_id}': {e}"
                    ) from e
                
                elif error_strategy == ErrorStrategy.COLLECT:
                    # Add error result
                    results.append(EvalCaseResult.error(
                        eval_id=case.eval_id,
                        name=case.name,
                        error=str(e)
                    ))
                
                # SKIP: just continue
        
        return EvalReport(
            eval_set_id=eval_set.eval_set_id,
            results=results,
            errors=errors if error_strategy == ErrorStrategy.COLLECT else [],
            summary=self._compute_summary(results)
        )
    
    async def _evaluate_case_safe(self, case: EvalCase) -> EvalCaseResult:
        """Evaluate case with error wrapping."""
        try:
            return await self._evaluate_case(case)
        except GraphExecutionError as e:
            raise  # Re-raise with context
        except Exception as e:
            raise GraphExecutionError(
                f"Error executing graph for case '{case.eval_id}'",
                e
            ) from e
```

### 7.5 Performance Optimization Strategies

```python
# agentflow/evaluation/performance.py

class PerformanceOptimizer:
    """
    Performance optimization strategies for evaluation.
    
    Techniques:
    1. Caching LLM judge results
    2. Batching LLM API calls
    3. Parallel case execution
    4. Result streaming
    5. Early termination
    """
    
    @staticmethod
    def create_llm_cache(cache_dir: str = ".eval_cache") -> LLMCache:
        """Create LLM result cache."""
        return DiskLLMCache(cache_dir)
    
    @staticmethod
    def batch_llm_evaluations(
        evaluations: list[tuple[str, str]],
        model: str,
        batch_size: int = 10
    ) -> list[float]:
        """
        Batch multiple LLM evaluations into fewer API calls.
        
        Strategy: Pack multiple comparison tasks into one prompt
        """
        results = []
        
        for i in range(0, len(evaluations), batch_size):
            batch = evaluations[i:i + batch_size]
            
            # Create batched prompt
            prompt = _create_batch_prompt(batch)
            
            # Single API call
            response = await _call_llm(model, prompt)
            
            # Parse batch results
            batch_results = _parse_batch_results(response)
            results.extend(batch_results)
        
        return results
    
    @staticmethod
    async def stream_results(
        evaluator: AgentEvaluator,
        eval_set: EvalSet
    ) -> AsyncIterator[EvalCaseResult]:
        """
        Stream results as they complete instead of waiting for all.
        
        Useful for:
        - Early feedback
        - Progress tracking
        - Memory efficiency with large eval sets
        """
        async for case in eval_set.eval_cases:
            result = await evaluator._evaluate_case(case)
            yield result


# Caching implementation

class DiskLLMCache:
    """
    Disk-based cache for LLM judge results.
    
    Cache Key: hash(actual_response + expected_response + model + criterion)
    Cache Value: EvalResult
    
    Thread-safe: Uses file locking
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def get(
        self,
        actual: str,
        expected: str,
        model: str,
        criterion: str
    ) -> CriterionResult | None:
        """Get cached result."""
        key = self._compute_key(actual, expected, model, criterion)
        cache_file = self.cache_dir / f"{key}.json"
        
        if cache_file.exists():
            with portalocker.Lock(cache_file, "r") as f:
                data = json.load(f)
                return CriterionResult.model_validate(data)
        
        return None
    
    def set(
        self,
        actual: str,
        expected: str,
        model: str,
        criterion: str,
        result: CriterionResult
    ) -> None:
        """Cache result."""
        key = self._compute_key(actual, expected, model, criterion)
        cache_file = self.cache_dir / f"{key}.json"
        
        with portalocker.Lock(cache_file, "w") as f:
            json.dump(result.model_dump(), f)
    
    @staticmethod
    def _compute_key(actual: str, expected: str, model: str, criterion: str) -> str:
        """Compute cache key."""
        content = f"{actual}||{expected}||{model}||{criterion}"
        return hashlib.sha256(content.encode()).hexdigest()


# Parallel execution with semaphore

class ParallelExecutor:
    """
    Parallel case execution with concurrency control.
    
    Uses asyncio.Semaphore to limit concurrent evaluations.
    This prevents:
    - Rate limit errors
    - Memory exhaustion
    - Resource contention
    """
    
    def __init__(self, max_concurrency: int = 4):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.active_tasks: set[asyncio.Task] = set()
    
    async def execute_parallel(
        self,
        evaluator: AgentEvaluator,
        cases: list[EvalCase]
    ) -> list[EvalCaseResult]:
        """Execute cases in parallel with semaphore."""
        
        async def evaluate_with_semaphore(case: EvalCase) -> EvalCaseResult:
            async with self.semaphore:
                return await evaluator._evaluate_case(case)
        
        tasks = [
            asyncio.create_task(evaluate_with_semaphore(case))
            for case in cases
        ]
        
        self.active_tasks.update(tasks)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Convert exceptions to error results
            final_results = []
            for case, result in zip(cases, results):
                if isinstance(result, Exception):
                    final_results.append(EvalCaseResult.error(
                        eval_id=case.eval_id,
                        name=case.name,
                        error=str(result)
                    ))
                else:
                    final_results.append(result)
            
            return final_results
            
        finally:
            self.active_tasks.clear()
    
    def cancel_all(self) -> None:
        """Cancel all active tasks."""
        for task in self.active_tasks:
            task.cancel()
```

### 7.6 Extensibility Points

```python
# Developer guide for extending the evaluation system

"""
Extensibility Points:

1. Custom Criteria
   - Inherit from BaseCriterion or SyncCriterion
   - Implement evaluate() method
   - Register via config or evaluator.add_criterion()

2. Custom Reporters
   - Implement report() method
   - Access EvalReport structure
   - Output in any format

3. Custom Trajectory Steps
   - Extend StepType enum
   - Add handler in TrajectoryCollector._handle_*()
   - Process custom events

4. Custom Eval Set Loaders
   - Implement EvalSetLoader protocol
   - Support custom file formats
   - Register with EvalSet.register_loader()

5. Custom Graph Adapters
   - Implement GraphAdapter protocol
   - Handle framework-specific graph execution
   - Inject custom collection logic

6. Middleware/Plugins
   - Pre/post evaluation hooks
   - Criterion result transformation
   - Custom metrics collection
"""

# Example: Custom Criterion

class MyCustomCriterion(BaseCriterion):
    """
    Custom criterion template.
    
    Implementation checklist:
    [ ] Set name and description
    [ ] Implement evaluate() method
    [ ] Return CriterionResult with score
    [ ] Handle errors gracefully
    [ ] Add unit tests
    [ ] Document expected input/output
    """
    
    name = "my_custom_criterion"
    description = "Description of what this evaluates"
    
    def __init__(self, config: CriterionConfig | None = None):
        super().__init__(config)
        # Initialize any required resources
        self.my_resource = self._init_resource()
    
    async def evaluate(
        self,
        actual: TrajectoryCollector,
        expected: EvalCase,
        context: EvaluationContext
    ) -> CriterionResult:
        """
        Evaluate the test case.
        
        Args:
            actual: Collected trajectory from execution
            expected: Expected behavior from test case
            context: Evaluation context with shared state
        
        Returns:
            Result with score 0.0-1.0 and pass/fail status
        """
        try:
            # 1. Extract relevant data
            actual_data = self._extract_actual_data(actual)
            expected_data = self._extract_expected_data(expected)
            
            # 2. Compute score
            score = self._compute_score(actual_data, expected_data)
            
            # 3. Create result
            return CriterionResult(
                criterion=self.name,
                score=score,
                passed=score >= self.config.threshold,
                threshold=self.config.threshold,
                details={
                    "actual": actual_data,
                    "expected": expected_data,
                    # Add useful debugging info
                }
            )
            
        except Exception as e:
            # Handle errors gracefully
            context.logger.error(f"Error in {self.name}: {e}")
            return CriterionResult.error(
                criterion=self.name,
                error=str(e),
                threshold=self.config.threshold
            )
    
    def _extract_actual_data(self, actual: TrajectoryCollector) -> Any:
        """Extract relevant data from trajectory."""
        # Implementation specific to your criterion
        pass
    
    def _extract_expected_data(self, expected: EvalCase) -> Any:
        """Extract expected data from test case."""
        # Implementation specific to your criterion
        pass
    
    def _compute_score(self, actual: Any, expected: Any) -> float:
        """
        Compute similarity score.
        
        Returns:
            Float between 0.0 (no match) and 1.0 (perfect match)
        """
        # Implementation specific to your criterion
        pass


# Example: Custom Reporter

class MyCustomReporter:
    """
    Custom reporter template.
    
    Use cases:
    - Export to custom format
    - Send to external system
    - Generate specialized visualizations
    """
    
    def report(self, report: EvalReport) -> None:
        """Output the report."""
        # Access report data
        summary = report.summary
        results = report.results
        
        # Format and output
        formatted = self._format_report(report)
        self._output(formatted)
    
    def save(self, report: EvalReport, filepath: str) -> None:
        """Save report to file."""
        output = self._format_report(report)
        with open(filepath, "w") as f:
            f.write(output)
    
    def _format_report(self, report: EvalReport) -> str:
        """Format report for output."""
        # Your formatting logic
        pass
    
    def _output(self, formatted: str) -> None:
        """Output formatted report."""
        # Print, send to API, etc.
        pass
```

### 7.7 Testing Strategy

```python
# Testing the evaluation system itself

"""
Test Pyramid for Evaluation System:

1. Unit Tests (Fast, Isolated)
   - Individual criterion logic
   - Data model validation
   - Trajectory parsing
   - Score calculation

2. Integration Tests (Medium)
   - Criterion + Collector integration
   - Evaluator + Graph integration
   - CLI command execution
   - File I/O operations

3. End-to-End Tests (Slow)
   - Full evaluation runs
   - Multi-turn conversations
   - LLM-as-judge criteria
   - CLI workflows

4. Performance Tests
   - Large eval sets
   - Parallel execution
   - Memory usage
   - Cache effectiveness
"""

# Example unit test

import pytest
from agentflow.evaluation import TrajectoryMatchCriterion, ToolCall

class TestTrajectoryMatchCriterion:
    """Unit tests for trajectory matching."""
    
    def test_exact_match_success(self):
        """Test exact match with matching trajectories."""
        criterion = TrajectoryMatchCriterion(match_type="EXACT", threshold=1.0)
        
        actual = [
            ToolCall(name="get_weather", args={"city": "NYC"}),
            ToolCall(name="format_response", args={}),
        ]
        
        expected = [
            ToolCall(name="get_weather", args={"city": "NYC"}),
            ToolCall(name="format_response", args={}),
        ]
        
        result = criterion._compute_score(actual, expected)
        assert result == 1.0
    
    def test_exact_match_different_order(self):
        """Test exact match fails with different order."""
        criterion = TrajectoryMatchCriterion(match_type="EXACT", threshold=1.0)
        
        actual = [
            ToolCall(name="format_response", args={}),
            ToolCall(name="get_weather", args={"city": "NYC"}),
        ]
        
        expected = [
            ToolCall(name="get_weather", args={"city": "NYC"}),
            ToolCall(name="format_response", args={}),
        ]
        
        result = criterion._compute_score(actual, expected)
        assert result < 1.0
    
    def test_in_order_match_with_extras(self):
        """Test in-order match allows extra tools."""
        criterion = TrajectoryMatchCriterion(match_type="IN_ORDER", threshold=0.8)
        
        actual = [
            ToolCall(name="get_weather", args={"city": "NYC"}),
            ToolCall(name="log_event", args={}),  # Extra
            ToolCall(name="format_response", args={}),
        ]
        
        expected = [
            ToolCall(name="get_weather", args={"city": "NYC"}),
            ToolCall(name="format_response", args={}),
        ]
        
        result = criterion._compute_score(actual, expected)
        assert result == 1.0  # All expected tools present in order

# Example integration test

@pytest.mark.asyncio
async def test_evaluator_with_mock_graph():
    """Integration test with mock graph."""
    from unittest.mock import AsyncMock, MagicMock
    
    # Create mock graph
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(return_value=MagicMock(
        response="The weather is sunny",
        messages=[]
    ))
    
    # Create eval set
    eval_set = EvalSet(
        eval_set_id="test",
        name="Test",
        eval_cases=[
            EvalCase(
                eval_id="test_1",
                conversation=[
                    Invocation(
                        invocation_id="1",
                        user_content=MessageContent.user("What's the weather?"),
                        expected_final_response=MessageContent.assistant("The weather is sunny")
                    )
                ]
            )
        ]
    )
    
    # Run evaluation
    evaluator = AgentEvaluator(graph, EvalConfig.default())
    report = await evaluator.evaluate(eval_set)
    
    # Assert
    assert report.summary.total_cases == 1
    assert report.summary.pass_rate > 0

# Example E2E test

@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_weather_agent_evaluation():
    """End-to-end test with real weather agent."""
    from examples.weather_agent import create_weather_agent
    
    # Create real agent
    graph = await create_weather_agent()
    
    # Load real eval set
    eval_set = EvalSet.load("tests/fixtures/weather_agent.evalset.json")
    
    # Run full evaluation
    evaluator = AgentEvaluator(graph, EvalConfig.default())
    report = await evaluator.evaluate(eval_set)
    
    # Assertions
    assert report.summary.pass_rate >= 0.8
    assert report.summary.total_cases > 0
    
    # Check specific criteria
    assert "trajectory_match" in report.summary.criterion_stats
    assert "response_match" in report.summary.criterion_stats
```

## 8. Action Plan (Developer Implementation Roadmap)

### Phase 1: Foundation & Data Models (Week 1-2)

**Module Structure Setup**
- [ ] Create `agentflow/evaluation/` package structure
- [ ] Set up `__init__.py` with proper exports
- [ ] Create `errors.py` for custom exceptions
- [ ] Set up `types.py` for type aliases and protocols

**Core Data Models** (`eval_set.py`, `eval_result.py`)
- [ ] Implement `EvalCase` with Pydantic validation
  - [ ] Add `conversation` field with proper Message types
  - [ ] Add `session_input` support
  - [ ] Implement `tags` and `metadata` fields
  - [ ] Add `.load()` and `.save()` class methods
- [ ] Implement `EvalSet` container
  - [ ] Add case collection management
  - [ ] Implement filtering by tags
  - [ ] Add validation logic
  - [ ] Implement merge functionality
- [ ] Implement `Invocation` model
  - [ ] Support multi-turn conversations
  - [ ] Add expected trajectory fields
  - [ ] Add expected response fields
- [ ] Implement result models
  - [ ] `EvalReport` with summary statistics
  - [ ] `EvalCaseResult` for individual cases
  - [ ] `CriterionResult` for criterion scores
  - [ ] `EvalSummary` with aggregated metrics

**Configuration System** (`eval_config.py`)
- [ ] Implement `EvalConfig` with criterion configuration
- [ ] Implement `CriterionConfig` with thresholds
- [ ] Add `MatchType` enum (EXACT, IN_ORDER, ANY_ORDER)
- [ ] Implement `RetryConfig` for error handling
- [ ] Add `.default()` factory methods
- [ ] Add JSON schema generation

**Testing for Phase 1**
- [ ] Unit tests for all data models
- [ ] Test JSON serialization/deserialization
- [ ] Test validation logic
- [ ] Test edge cases (empty sets, invalid data)

### Phase 2: Trajectory Collection (Week 2-3)

**Event Integration** (`collectors/trajectory_collector.py`)
- [ ] Implement `TrajectoryCollector` base class
  - [ ] Subscribe to `EventModel` stream
  - [ ] Implement `on_event()` async callback
  - [ ] Add event filtering logic
  - [ ] Build trajectory in real-time
- [ ] Implement event handlers
  - [ ] `_handle_node_event()` for node execution
  - [ ] `_handle_tool_event()` for tool calls
  - [ ] `_handle_message_event()` for LLM outputs
  - [ ] `_handle_conditional_event()` for routing
- [ ] Implement state correlation
  - [ ] Match START/END events by ID
  - [ ] Track pending tool calls
  - [ ] Calculate timing metrics
  - [ ] Handle out-of-order events
- [ ] Add `get_tool_calls()` extraction method
- [ ] Add `get_trajectory()` with filtering

**Trajectory Data Structures** (`collectors/trajectory.py`)
- [ ] Implement `TrajectoryStep` model
- [ ] Implement `StepType` enum
- [ ] Implement `ToolCall` model with result tracking
- [ ] Add helper methods for trajectory analysis

**Graph Adapter** (`adapters/graph_adapter.py`)
- [ ] Implement `GraphEvaluationAdapter`
  - [ ] Inject collector into callback chain
  - [ ] Handle multi-turn state management
  - [ ] Extract responses from results
  - [ ] Clean up resources properly
- [ ] Add support for different graph types
- [ ] Implement state transformation utilities

**Testing for Phase 2**
- [ ] Unit tests for collector logic
- [ ] Integration tests with mock events
- [ ] Test event correlation logic
- [ ] Test with sample graph executions

### Phase 3: Core Criteria Implementation (Week 3-4)

**Base Criterion Infrastructure** (`criteria/base.py`)
- [ ] Implement `BaseCriterion` abstract class
  - [ ] Define `evaluate()` method signature
  - [ ] Add config property
  - [ ] Add name and description
- [ ] Implement `SyncCriterion` for non-async criteria
- [ ] Implement `CriterionResult` builder methods
- [ ] Add criterion registry system

**Trajectory Criteria** (`criteria/trajectory.py`)
- [ ] Implement `TrajectoryMatchCriterion`
  - [ ] EXACT matching mode
  - [ ] IN_ORDER matching mode
  - [ ] ANY_ORDER matching mode
  - [ ] Argument comparison logic
  - [ ] Partial credit calculation
- [ ] Implement `ToolNameMatchCriterion`
- [ ] Implement `NodeSequenceCriterion`

**Response Criteria** (`criteria/response.py`)
- [ ] Implement `ResponseMatchCriterion`
  - [ ] ROUGE-1 calculation
  - [ ] Precision, recall, F1
- [ ] Implement `ExactMatchCriterion`
- [ ] Implement `ContainsKeywordsCriterion`
- [ ] Implement `RegexMatchCriterion`

**LLM Judge Criteria** (`criteria/llm_judge.py`)
- [ ] Implement `LLMJudgeCriterion`
  - [ ] Prompt template management
  - [ ] Multi-sample voting
  - [ ] Result parsing with retries
  - [ ] Error handling for API failures
- [ ] Implement caching layer
  - [ ] Hash-based cache keys
  - [ ] Disk persistence
  - [ ] Cache invalidation
- [ ] Add rate limiting support
- [ ] Implement batch API calls

**Testing for Phase 3**
- [ ] Unit tests for each criterion
- [ ] Test all matching modes
- [ ] Test edge cases (empty, malformed data)
- [ ] Integration tests with mock LLM responses
- [ ] Performance tests for caching

### Phase 4: Evaluator & Pipeline (Week 4-5)

**Evaluation Pipeline** (`evaluator.py`)
- [ ] Implement `AgentEvaluator` class
  - [ ] Initialize with graph and config
  - [ ] Build criterion list from config
  - [ ] Implement `evaluate()` method
  - [ ] Support both EvalSet and file paths
- [ ] Implement `EvaluationPipeline`
  - [ ] Sequential execution mode
  - [ ] Parallel execution mode
  - [ ] Retry logic with backoff
  - [ ] Timeout handling
  - [ ] Early termination (fail-fast)
- [ ] Implement `EvaluationContext`
  - [ ] Logger integration
  - [ ] Metrics collection
  - [ ] Shared state management
  - [ ] Cancel token support

**Error Handling** (`error_handling.py`)
- [ ] Implement error strategy enum
  - [ ] FAIL_FAST mode
  - [ ] COLLECT mode
  - [ ] SKIP mode
- [ ] Implement error wrapping
- [ ] Add error recovery logic
- [ ] Implement error reporting

**Progress Tracking** (`progress.py`)
- [ ] Implement progress callbacks
- [ ] Add streaming result support
- [ ] Implement event hooks
  - [ ] `on_case_start`
  - [ ] `on_case_complete`
  - [ ] `on_criterion_complete`

**Testing for Phase 4**
- [ ] Integration tests with full pipeline
- [ ] Test error handling strategies
- [ ] Test parallel execution
- [ ] Test progress callbacks
- [ ] Memory leak tests

### Phase 5: Reporters & Output (Week 5)

**Reporter Infrastructure** (`reporters/base.py`)
- [ ] Define reporter protocol
- [ ] Implement base formatter utilities

**Console Reporter** (`reporters/console.py`)
- [ ] Implement `ConsoleReporter` with Rich
  - [ ] Summary table
  - [ ] Criterion breakdown
  - [ ] Failed case details
  - [ ] Color-coded output
- [ ] Add verbose mode
- [ ] Implement progress bars

**File Reporters** (`reporters/json.py`, `reporters/xml.py`)
- [ ] Implement `JSONReporter`
  - [ ] Full report export
  - [ ] Filtered export (failures only)
  - [ ] Pretty printing
- [ ] Implement `JUnitXMLReporter`
  - [ ] Standard JUnit format
  - [ ] CI/CD compatibility
- [ ] Implement `HTMLReporter`
  - [ ] Interactive report generation
  - [ ] Charts and visualizations
  - [ ] Filtering and search

**Testing for Phase 5**
- [ ] Test each reporter format
- [ ] Validate output structure
- [ ] Test with various report sizes

### Phase 6: Advanced Features (Week 6-7)

**Rubric-Based Evaluation** (`criteria/rubric.py`)
- [ ] Implement `RubricBasedCriterion`
  - [ ] Rubric model with scoring guide
  - [ ] Multi-dimensional scoring
  - [ ] Weighted aggregation
- [ ] Add rubric templates

**Safety & Quality Criteria** (`criteria/advanced.py`)
- [ ] Implement `HallucinationCriterion`
  - [ ] Groundedness checking
  - [ ] Claim extraction
  - [ ] Context verification
- [ ] Implement `SafetyCriterion`
  - [ ] Multiple safety categories
  - [ ] Threshold configuration per category
- [ ] Implement `FactualAccuracyCriterion`

**User Simulation** (`simulators/user_simulator.py`)
- [ ] Implement `ConversationScenario` model
- [ ] Implement `UserSimulator`
  - [ ] Prompt generation
  - [ ] Goal tracking
  - [ ] Conversation state management
- [ ] Implement `BatchSimulator`
  - [ ] Parallel scenario execution
  - [ ] Result aggregation
- [ ] Implement `SimulationResult` model

**Testing for Phase 6**
- [ ] Test rubric evaluation
- [ ] Test safety criteria
- [ ] Test user simulation
- [ ] End-to-end simulation tests

### Phase 7: CLI Implementation (Week 7-8)

**CLI Infrastructure** (`cli/main.py`)
- [ ] Set up Click application
- [ ] Implement command groups
  - [ ] `agentflow eval` group
  - [ ] `agentflow eval-set` group
  - [ ] `agentflow simulate` group
- [ ] Add global options (verbose, config, etc.)

**Eval Commands** (`cli/eval_commands.py`)
- [ ] Implement `eval run` command
  - [ ] Agent loading logic
  - [ ] Config loading
  - [ ] Progress tracking with Rich
  - [ ] Multi-format output
- [ ] Implement `eval watch` command
  - [ ] File watching with watchdog
  - [ ] Auto-rerun on changes
- [ ] Implement `eval compare` command
  - [ ] Report comparison
  - [ ] Regression detection

**EvalSet Commands** (`cli/evalset_commands.py`)
- [ ] Implement `eval-set create`
- [ ] Implement `eval-set add`
  - [ ] From session
  - [ ] From template
  - [ ] Interactive mode
- [ ] Implement `eval-set list`
- [ ] Implement `eval-set validate`
- [ ] Implement `eval-set merge`

**Simulation Commands** (`cli/simulate_commands.py`)
- [ ] Implement `simulate run`
- [ ] Implement `simulate batch`

**CLI Utilities** (`cli/utils.py`, `cli/formatters.py`)
- [ ] Agent module loading
- [ ] Output formatting
- [ ] Error handling and display

**Testing for Phase 7**
- [ ] CLI integration tests
- [ ] Test each command
- [ ] Test error scenarios
- [ ] Test with sample projects

### Phase 8: Testing & Quality (Week 8-9)

**Comprehensive Test Suite**
- [ ] Unit test coverage > 90%
- [ ] Integration test suite
- [ ] End-to-end test scenarios
- [ ] Performance benchmarks
- [ ] Memory profiling

**Test Infrastructure**
- [ ] Pytest fixtures for common setups
- [ ] Mock implementations
- [ ] Test data generators
- [ ] CI/CD pipeline integration

**Quality Checks**
- [ ] Type checking with mypy
- [ ] Linting with ruff
- [ ] Code formatting with black
- [ ] Documentation coverage

### Phase 9: Documentation (Week 9-10)

**Developer Documentation**
- [ ] Architecture overview
- [ ] API reference (auto-generated)
- [ ] Integration guide
- [ ] Extension guide (custom criteria, reporters)
- [ ] Performance tuning guide

**User Documentation**
- [ ] Getting started tutorial
- [ ] Evaluation guide
- [ ] CLI reference
- [ ] Best practices guide
- [ ] Example gallery

**Examples**
- [ ] Weather agent example
- [ ] Multi-turn conversation example
- [ ] Custom criterion example
- [ ] CI/CD integration example
- [ ] Advanced scenarios

### Phase 10: Polish & Release (Week 10)

**Performance Optimization**
- [ ] Profile and optimize hot paths
- [ ] Implement caching strategies
- [ ] Optimize parallel execution
- [ ] Reduce memory footprint

**Final Testing**
- [ ] Load testing with large eval sets
- [ ] Stress testing parallel execution
- [ ] Cross-platform testing
- [ ] Real-world agent testing

**Release Preparation**
- [ ] Version numbering
- [ ] Changelog
- [ ] Migration guide (if needed)
- [ ] Release notes

---

## 8. Example Eval Set File

```json
{
  "eval_set_id": "weather_agent_basic_tests",
  "name": "Weather Agent Basic Tests",
  "description": "Basic functionality tests for the weather agent",
  "eval_cases": [
    {
      "eval_id": "weather_lookup_simple",
      "conversation": [
        {
          "invocation_id": "inv_001",
          "user_content": {
            "role": "user",
            "content": [{"type": "text", "text": "What's the weather in New York?"}]
          },
          "expected_tool_trajectory": [
            {
              "name": "get_weather",
              "args": {"location": "New York"}
            }
          ],
          "expected_final_response": {
            "role": "assistant",
            "content": [{"type": "text", "text": "The weather in New York is currently sunny with a temperature of 72°F."}]
          }
        }
      ],
      "session_input": {
        "app_name": "weather_agent",
        "user_id": "test_user"
      }
    },
    {
      "eval_id": "weather_lookup_multi_city",
      "conversation": [
        {
          "invocation_id": "inv_002",
          "user_content": {
            "role": "user",
            "content": [{"type": "text", "text": "Compare the weather in Tokyo and London"}]
          },
          "expected_tool_trajectory": [
            {"name": "get_weather", "args": {"location": "Tokyo"}},
            {"name": "get_weather", "args": {"location": "London"}}
          ],
          "expected_final_response": {
            "role": "assistant", 
            "content": [{"type": "text", "text": "Tokyo is sunny at 75°F while London is cloudy at 55°F."}]
          }
        }
      ],
      "session_input": {
        "app_name": "weather_agent",
        "user_id": "test_user"
      }
    }
  ]
}
```

---

## 9. Example Eval Config File

```json
{
  "criteria": {
    "tool_trajectory_avg_score": {
      "threshold": 1.0,
      "match_type": "IN_ORDER"
    },
    "response_match_score": {
      "threshold": 0.7
    },
    "final_response_match_v2": {
      "threshold": 0.8,
      "judge_model": "gpt-4o-mini",
      "num_samples": 3
    }
  },
  "user_simulator_config": {
    "model": "gpt-4o",
    "max_allowed_invocations": 10
  }
}
```

---

## 10. Benefits Summary

| Benefit | Description |
|---------|-------------|
| **Regression Testing** | Ensure agent updates don't break existing functionality |
| **CI/CD Integration** | Automated testing in deployment pipelines |
| **Trajectory Validation** | Verify agents follow expected tool usage patterns |
| **Response Quality** | Measure semantic correctness of agent outputs |
| **Debugging** | Identify specific failure points in agent logic |
| **Partial Credit** | Get nuanced scores instead of binary pass/fail |
| **Dynamic Testing** | AI-powered user simulation for realistic scenarios |
| **Custom Rubrics** | Define domain-specific quality criteria |

---

## 11. References

- [Google ADK Evaluation Documentation](https://google.github.io/adk-docs/evaluate/)
- [Google ADK Evaluation Criteria](https://google.github.io/adk-docs/evaluate/criteria/)
- [LangSmith Agent Evaluation](https://docs.langchain.com/langsmith/evaluate-complex-agent)
- [LangSmith Evaluation Concepts](https://docs.langchain.com/langsmith/evaluation-concepts)

---

## Appendix A: Comparison Matrix

| Feature | Google ADK | LangSmith | Agentflow (Proposed) |
|---------|------------|-----------|----------------------|
| Trajectory Matching | ✅ EXACT/IN_ORDER/ANY_ORDER | ✅ Subsequence | ✅ All three modes |
| Response Matching | ✅ ROUGE-1 | ✅ LLM-as-judge | ✅ Both |
| LLM Judge | ✅ | ✅ | ✅ |
| Custom Rubrics | ✅ | ❌ (manual) | ✅ |
| Hallucination Detection | ✅ | ❌ | ✅ |
| Safety Evaluation | ✅ | ❌ | ✅ |
| User Simulation | ✅ | ❌ | ✅ |
| Single Step Testing | ❌ | ✅ | ✅ |
| CLI Support | ✅ | ✅ (via Python) | ✅ |
| pytest Integration | ✅ | ✅ | ✅ |
| Web UI | ✅ | ✅ (LangSmith) | Future |
