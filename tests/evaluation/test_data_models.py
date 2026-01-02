"""
Tests for the evaluation module data models.

Tests for EvalSet, EvalCase, EvalConfig, and related models.
"""

import json
import tempfile
from pathlib import Path

import pytest

from agentflow.evaluation import (
    CriterionConfig,
    EvalCase,
    EvalConfig,
    EvalSet,
    Invocation,
    MatchType,
    MessageContent,
    Rubric,
    SessionInput,
    StepType,
    ToolCall,
    TrajectoryStep,
)


class TestToolCall:
    """Tests for ToolCall model."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        tc = ToolCall(name="get_weather", args={"location": "NYC"})
        assert tc.name == "get_weather"
        assert tc.args == {"location": "NYC"}
        assert tc.call_id is None
        assert tc.result is None

    def test_tool_call_with_all_fields(self):
        """Test tool call with all fields."""
        tc = ToolCall(
            name="search",
            args={"query": "test"},
            call_id="call_123",
            result={"results": ["a", "b"]},
        )
        assert tc.name == "search"
        assert tc.call_id == "call_123"
        assert tc.result == {"results": ["a", "b"]}

    def test_tool_call_matches_same(self):
        """Test matching identical tool calls."""
        tc1 = ToolCall(name="get_weather", args={"location": "NYC"})
        tc2 = ToolCall(name="get_weather", args={"location": "NYC"})
        assert tc1.matches(tc2)
        assert tc1 == tc2

    def test_tool_call_matches_different_name(self):
        """Test matching tool calls with different names."""
        tc1 = ToolCall(name="get_weather", args={"location": "NYC"})
        tc2 = ToolCall(name="get_time", args={"location": "NYC"})
        assert not tc1.matches(tc2)
        assert tc1 != tc2

    def test_tool_call_matches_different_args(self):
        """Test matching tool calls with different args."""
        tc1 = ToolCall(name="get_weather", args={"location": "NYC"})
        tc2 = ToolCall(name="get_weather", args={"location": "LA"})
        assert not tc1.matches(tc2, check_args=True)
        assert tc1.matches(tc2, check_args=False)

    def test_tool_call_matches_with_call_id(self):
        """Test matching tool calls checking call_id."""
        tc1 = ToolCall(name="get_weather", args={}, call_id="call_1")
        tc2 = ToolCall(name="get_weather", args={}, call_id="call_2")
        assert tc1.matches(tc2, check_call_id=False)
        assert not tc1.matches(tc2, check_call_id=True)


class TestTrajectoryStep:
    """Tests for TrajectoryStep model."""

    def test_trajectory_step_node_factory(self):
        """Test creating node step via factory method."""
        step = TrajectoryStep.node("agent_node", timestamp=1234567890.0)
        assert step.step_type == StepType.NODE
        assert step.name == "agent_node"
        assert step.timestamp == 1234567890.0
        assert step.args == {}

    def test_trajectory_step_tool_factory(self):
        """Test creating tool step via factory method."""
        step = TrajectoryStep.tool(
            "get_weather",
            args={"location": "NYC"},
            timestamp=1234567890.0,
            tool_call_id="call_123",
        )
        assert step.step_type == StepType.TOOL
        assert step.name == "get_weather"
        assert step.args == {"location": "NYC"}
        assert step.metadata["tool_call_id"] == "call_123"

    def test_trajectory_step_direct_creation(self):
        """Test creating step directly."""
        step = TrajectoryStep(
            step_type=StepType.MESSAGE,
            name="assistant_response",
            metadata={"content": "Hello!"},
        )
        assert step.step_type == StepType.MESSAGE
        assert step.name == "assistant_response"


class TestMessageContent:
    """Tests for MessageContent model."""

    def test_message_content_user_factory(self):
        """Test creating user message."""
        msg = MessageContent.user("What's the weather?")
        assert msg.role == "user"
        assert msg.content == "What's the weather?"

    def test_message_content_assistant_factory(self):
        """Test creating assistant message."""
        msg = MessageContent.assistant("The weather is sunny.")
        assert msg.role == "assistant"
        assert msg.content == "The weather is sunny."

    def test_message_content_get_text_string(self):
        """Test extracting text from string content."""
        msg = MessageContent(role="user", content="Hello world")
        assert msg.get_text() == "Hello world"

    def test_message_content_get_text_blocks(self):
        """Test extracting text from content blocks."""
        msg = MessageContent(
            role="user",
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        )
        assert msg.get_text() == "Hello World"


class TestInvocation:
    """Tests for Invocation model."""

    def test_invocation_simple_factory(self):
        """Test creating invocation via simple factory."""
        inv = Invocation.simple(
            user_query="What's the weather?",
            expected_response="It's sunny.",
            expected_tools=[ToolCall(name="get_weather", args={"location": "NYC"})],
        )
        assert inv.user_content.get_text() == "What's the weather?"
        assert inv.expected_final_response is not None
        assert inv.expected_final_response.get_text() == "It's sunny."
        assert len(inv.expected_tool_trajectory) == 1
        assert inv.expected_tool_trajectory[0].name == "get_weather"

    def test_invocation_has_uuid(self):
        """Test invocation generates UUID by default."""
        inv = Invocation(user_content=MessageContent.user("Hello"))
        assert inv.invocation_id is not None
        assert len(inv.invocation_id) > 0


class TestSessionInput:
    """Tests for SessionInput model."""

    def test_session_input_defaults(self):
        """Test session input default values."""
        session = SessionInput()
        assert session.app_name == ""
        assert session.user_id == "test_user"
        assert session.state == {}
        assert session.config == {}

    def test_session_input_with_values(self):
        """Test session input with custom values."""
        session = SessionInput(
            app_name="weather_agent",
            user_id="user_123",
            state={"location": "NYC"},
            config={"temperature": 0.7},
        )
        assert session.app_name == "weather_agent"
        assert session.user_id == "user_123"
        assert session.state["location"] == "NYC"


class TestEvalCase:
    """Tests for EvalCase model."""

    def test_eval_case_single_turn_factory(self):
        """Test creating single-turn eval case."""
        case = EvalCase.single_turn(
            eval_id="test_001",
            user_query="What's the weather in NYC?",
            expected_response="It's sunny in NYC.",
            expected_tools=[ToolCall(name="get_weather", args={"location": "NYC"})],
            name="Weather Query Test",
        )
        assert case.eval_id == "test_001"
        assert case.name == "Weather Query Test"
        assert len(case.conversation) == 1
        assert case.conversation[0].user_content.get_text() == "What's the weather in NYC?"

    def test_eval_case_multi_turn(self):
        """Test creating multi-turn eval case."""
        case = EvalCase(
            eval_id="test_002",
            name="Multi-turn Test",
            conversation=[
                Invocation.simple("Hello"),
                Invocation.simple("What's the weather?", "It's sunny."),
            ],
            tags=["weather", "multi-turn"],
        )
        assert len(case.conversation) == 2
        assert "weather" in case.tags
        assert "multi-turn" in case.tags


class TestEvalSet:
    """Tests for EvalSet model."""

    def test_eval_set_creation(self):
        """Test creating eval set."""
        eval_set = EvalSet(
            eval_set_id="weather_tests",
            name="Weather Agent Tests",
            description="Tests for weather agent functionality",
        )
        assert eval_set.eval_set_id == "weather_tests"
        assert eval_set.name == "Weather Agent Tests"
        assert len(eval_set) == 0

    def test_eval_set_add_case(self):
        """Test adding cases to eval set."""
        eval_set = EvalSet(eval_set_id="tests")
        case = EvalCase.single_turn("test_001", "Hello", "Hi there!")
        eval_set.add_case(case)
        assert len(eval_set) == 1
        assert eval_set.get_case("test_001") == case

    def test_eval_set_iteration(self):
        """Test iterating over eval set."""
        eval_set = EvalSet(eval_set_id="tests")
        for i in range(3):
            eval_set.add_case(EvalCase.single_turn(f"test_{i}", f"Query {i}"))
        
        ids = [case.eval_id for case in eval_set]
        assert ids == ["test_0", "test_1", "test_2"]

    def test_eval_set_filter_by_tags(self):
        """Test filtering cases by tags."""
        eval_set = EvalSet(eval_set_id="tests")
        eval_set.add_case(EvalCase(eval_id="1", tags=["weather", "basic"]))
        eval_set.add_case(EvalCase(eval_id="2", tags=["weather", "advanced"]))
        eval_set.add_case(EvalCase(eval_id="3", tags=["search"]))
        
        weather_cases = eval_set.filter_by_tags(["weather"])
        assert len(weather_cases) == 2
        
        advanced_weather = eval_set.filter_by_tags(["weather", "advanced"])
        assert len(advanced_weather) == 1
        assert advanced_weather[0].eval_id == "2"

    def test_eval_set_serialization(self):
        """Test saving and loading eval set to/from file."""
        eval_set = EvalSet(
            eval_set_id="test_set",
            name="Test Set",
            eval_cases=[
                EvalCase.single_turn("test_001", "Hello", "Hi!"),
            ],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_set.json"
            eval_set.to_file(str(path))
            
            # Verify file was created
            assert path.exists()
            
            # Load back
            loaded = EvalSet.from_file(str(path))
            assert loaded.eval_set_id == eval_set.eval_set_id
            assert loaded.name == eval_set.name
            assert len(loaded) == 1
            assert loaded.get_case("test_001") is not None


class TestCriterionConfig:
    """Tests for CriterionConfig model."""

    def test_criterion_config_defaults(self):
        """Test default criterion config."""
        config = CriterionConfig()
        assert config.threshold == 0.8
        assert config.match_type == MatchType.EXACT
        assert config.enabled is True

    def test_criterion_config_trajectory_factory(self):
        """Test trajectory config factory."""
        config = CriterionConfig.trajectory(
            threshold=0.9,
            match_type=MatchType.IN_ORDER,
            check_args=False,
        )
        assert config.threshold == 0.9
        assert config.match_type == MatchType.IN_ORDER
        assert config.check_args is False

    def test_criterion_config_llm_judge_factory(self):
        """Test LLM judge config factory."""
        config = CriterionConfig.llm_judge(
            threshold=0.85,
            judge_model="gpt-4",
            num_samples=5,
        )
        assert config.threshold == 0.85
        assert config.judge_model == "gpt-4"
        assert config.num_samples == 5

    def test_criterion_config_rubric_factory(self):
        """Test rubric config factory."""
        rubrics = [
            Rubric(rubric_id="concise", content="Response is concise."),
            Rubric(rubric_id="helpful", content="Response is helpful."),
        ]
        config = CriterionConfig.rubric_based(rubrics=rubrics, threshold=0.75)
        assert config.threshold == 0.75
        assert len(config.rubrics) == 2


class TestEvalConfig:
    """Tests for EvalConfig model."""

    def test_eval_config_default(self):
        """Test default eval config."""
        config = EvalConfig.default()
        assert "tool_trajectory_avg_score" in config.criteria
        assert "response_match_score" in config.criteria
        assert config.criteria["tool_trajectory_avg_score"].threshold == 1.0
        assert config.criteria["response_match_score"].threshold == 0.8

    def test_eval_config_strict(self):
        """Test strict eval config."""
        config = EvalConfig.strict()
        assert config.criteria["tool_trajectory_avg_score"].threshold == 1.0
        assert config.criteria["response_match_score"].threshold == 0.9
        assert "final_response_match_v2" in config.criteria

    def test_eval_config_relaxed(self):
        """Test relaxed eval config."""
        config = EvalConfig.relaxed()
        assert config.criteria["tool_trajectory_avg_score"].threshold == 0.8
        assert config.criteria["tool_trajectory_avg_score"].match_type == MatchType.IN_ORDER
        assert config.criteria["response_match_score"].threshold == 0.6

    def test_eval_config_enable_criterion(self):
        """Test enabling a criterion."""
        config = EvalConfig()
        config.enable_criterion(
            "custom_criterion",
            CriterionConfig(threshold=0.7),
        )
        assert "custom_criterion" in config.criteria
        assert config.criteria["custom_criterion"].threshold == 0.7

    def test_eval_config_disable_criterion(self):
        """Test disabling a criterion."""
        config = EvalConfig.default()
        config.disable_criterion("tool_trajectory_avg_score")
        assert config.criteria["tool_trajectory_avg_score"].enabled is False

    def test_eval_config_with_rubrics(self):
        """Test adding rubrics to config."""
        config = EvalConfig.default()
        rubrics = [Rubric(rubric_id="test", content="Test rubric")]
        new_config = config.with_rubrics(rubrics)
        
        assert "rubric_based_quality" in new_config.criteria
        assert len(new_config.criteria["rubric_based_quality"].rubrics) == 1
        # Original should be unchanged
        assert "rubric_based_quality" not in config.criteria

    def test_eval_config_serialization(self):
        """Test saving and loading eval config."""
        config = EvalConfig.default()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            config.to_file(str(path))
            
            loaded = EvalConfig.from_file(str(path))
            assert "tool_trajectory_avg_score" in loaded.criteria
            assert loaded.criteria["tool_trajectory_avg_score"].threshold == 1.0


class TestRubric:
    """Tests for Rubric model."""

    def test_rubric_creation(self):
        """Test creating a rubric."""
        rubric = Rubric(
            rubric_id="conciseness",
            content="The response is concise and to the point.",
            weight=1.5,
        )
        assert rubric.rubric_id == "conciseness"
        assert rubric.weight == 1.5

    def test_rubric_create_factory(self):
        """Test rubric create factory."""
        rubric = Rubric.create(
            rubric_id="helpful",
            content="Response is helpful.",
            weight=2.0,
        )
        assert rubric.rubric_id == "helpful"
        assert rubric.weight == 2.0
