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

## 6. CLI Integration

```bash
# Run evaluation via CLI
agentflow eval \
    examples/react \
    tests/fixtures/weather_agent.evalset.json \
    --config tests/fixtures/eval_config.json \
    --output results/eval_report.json \
    --verbose

# Create new eval set from recorded sessions
agentflow eval-set create \
    examples/react \
    my_weather_tests

# Add test case to eval set
agentflow eval-set add \
    examples/react \
    my_weather_tests \
    --from-session session_12345.json
```

---

## 7. Action Plan

### Phase 1: Foundation (Week 1-2)
- [ ] Create `agentflow/evaluation/` module structure
- [ ] Implement `EvalSet`, `EvalCase`, `EvalConfig` data models
- [ ] Implement `TrajectoryCollector` using existing `EventModel` infrastructure
- [ ] Create base `BaseCriterion` interface

### Phase 2: Core Criteria (Week 2-3)
- [ ] Implement `TrajectoryMatchCriterion` (EXACT, IN_ORDER, ANY_ORDER)
- [ ] Implement `ResponseMatchCriterion` (ROUGE-1 based)
- [ ] Implement `LLMJudgeCriterion` for semantic matching
- [ ] Add unit tests for all criteria

### Phase 3: Evaluator (Week 3-4)
- [ ] Implement main `AgentEvaluator` class
- [ ] Add eval set loading from JSON files
- [ ] Implement evaluation reporting (console, JSON, HTML)
- [ ] Add pytest integration utilities

### Phase 4: Advanced Features (Week 4-5)
- [ ] Implement `RubricCriterion` for custom rubrics
- [ ] Add `HallucinationCriterion` for groundedness checking
- [ ] Implement `SafetyCriterion` for safety evaluation
- [ ] Add `UserSimulator` for dynamic conversation testing

### Phase 5: CLI & Documentation (Week 5-6)
- [ ] Add CLI commands (`agentflow eval`, `agentflow eval-set`)
- [ ] Create comprehensive documentation
- [ ] Add example evaluation files for existing examples
- [ ] Write tutorial on writing effective eval sets

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
