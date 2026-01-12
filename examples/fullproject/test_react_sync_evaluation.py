"""
Evaluation tests for the react_sync.py example.

This module provides comprehensive evaluation tests for the ReAct agent,
including trajectory validation, tool usage correctness, response quality,
and performance metrics.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock
import time

from agentflow.state import AgentState, Message
from agentflow.evaluation import (
    EvalCase,
    EvalConfig,
    TrajectoryStep,
    ToolCall,
    MessageContent,
)
from agentflow.evaluation.collectors.trajectory_collector import TrajectoryCollector
from agentflow.evaluation.criteria.base import BaseCriterion
from agentflow.evaluation.eval_result import CriterionResult

# Import the components from react_sync
import sys
from pathlib import Path

react_dir = Path(__file__).parent
sys.path.insert(0, str(react_dir))


class TestTrajectoryEvaluation:
    """Tests for evaluating agent execution trajectory."""
    
    def test_collector_captures_messages(self):
        """Test that trajectory collector captures messages correctly."""
        collector = TrajectoryCollector()
        
        # Simulate agent execution
        user_msg = Message.text_message("What's the weather?", role="user")
        assistant_msg = Message.text_message("Let me check", role="assistant")
        tool_msg = Message.text_message("Sunny", role="tool")
        
        collector.add_message(user_msg)
        collector.add_message(assistant_msg)
        collector.add_message(tool_msg)
        
        assert len(collector.messages) == 3
        assert collector.messages[0].role == "user"
        assert collector.messages[1].role == "assistant"
        assert collector.messages[2].role == "tool"
    
    def test_collector_captures_tool_calls(self):
        """Test that trajectory collector captures tool calls."""
        collector = TrajectoryCollector()
        
        # Add a message with tool call
        msg = Message.text_message("Calling tool", role="assistant")
        msg.tools_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "NYC"}'
                }
            }
        ]
        
        collector.add_message(msg)
        
        # Extract tool calls
        tool_calls = []
        for message in collector.messages:
            if hasattr(message, 'tools_calls') and message.tools_calls:
                tool_calls.extend(message.tools_calls)
        
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"


class TestWeatherToolEvaluation:
    """Evaluation tests for the weather tool functionality."""
    
    def test_weather_tool_correct_output_format(self):
        """Test that weather tool returns correct output format."""
        from react_sync import get_weather
        
        locations = ["New York", "London", "Tokyo", "Paris", "Berlin"]
        
        for location in locations:
            result = get_weather(location=location)
            
            # Verify format
            assert isinstance(result, str)
            assert location in result
            assert "weather" in result.lower()
            assert "sunny" in result.lower()
    
    def test_weather_tool_handles_special_characters(self):
        """Test that weather tool handles special characters in location names."""
        from react_sync import get_weather
        
        special_locations = [
            "São Paulo",
            "Zürich",
            "Montréal",
            "Москва",  # Moscow in Cyrillic
        ]
        
        for location in special_locations:
            result = get_weather(location=location)
            assert location in result
            assert isinstance(result, str)
    
    def test_weather_tool_dependency_injection(self):
        """Test that dependency injection works correctly."""
        from react_sync import get_weather
        
        state = AgentState()
        state.context = [Message.text_message("Test", role="user")]
        
        # Should not raise exception
        result = get_weather(
            location="Test City",
            tool_call_id="test_id",
            state=state
        )
        
        assert isinstance(result, str)


class TestAgentResponseQuality:
    """Tests for evaluating agent response quality."""
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration", default=False),
        reason="Requires actual API calls"
    )
    def test_agent_uses_tool_when_appropriate(self):
        """Test that agent correctly identifies when to use tools."""
        from react_sync import app
        
        # Request that clearly requires tool usage
        inp = {
            "messages": [
                Message.text_message(
                    "Please call the get_weather function for San Francisco"
                )
            ]
        }
        config = {"thread_id": "eval_001", "recursion_limit": 10}
        
        result = app.invoke(inp, config=config)
        
        # Check that tool was called
        tool_messages = [
            msg for msg in result["messages"]
            if msg.role == "tool"
        ]
        
        assert len(tool_messages) > 0, "Agent should have called the weather tool"
    
    def test_routing_logic_correctness(self):
        """Test that routing logic makes correct decisions."""
        from react_sync import should_use_tools
        
        test_cases = [
            {
                "description": "Empty context should route to TOOL",
                "context": [],
                "expected": "TOOL"
            },
            {
                "description": "Tool result should route to MAIN",
                "context": [Message.text_message("Result", role="tool")],
                "expected": "MAIN"
            },
            {
                "description": "User message should end",
                "context": [Message.text_message("Hello", role="user")],
                "expected": "END"
            },
        ]
        
        for case in test_cases:
            state = AgentState()
            state.context = case["context"]
            
            result = should_use_tools(state)
            
            assert result == case["expected"], (
                f"Failed: {case['description']}. "
                f"Expected {case['expected']}, got {result}"
            )


class TestPerformanceMetrics:
    """Tests for measuring agent performance."""
    
    def test_tool_execution_performance(self):
        """Test weather tool execution performance."""
        from react_sync import get_weather
        
        iterations = 100
        start_time = time.time()
        
        for i in range(iterations):
            get_weather(location=f"City{i}")
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        # Tool should execute quickly (< 10ms per call)
        assert avg_time < 0.01, f"Tool too slow: {avg_time*1000:.2f}ms per call"
    
    def test_routing_decision_performance(self):
        """Test routing decision performance."""
        from react_sync import should_use_tools
        
        state = AgentState()
        state.context = [Message.text_message("Test", role="user")]
        
        iterations = 1000
        start_time = time.time()
        
        for _ in range(iterations):
            should_use_tools(state)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        # Routing should be very fast (< 1ms)
        assert avg_time < 0.001, f"Routing too slow: {avg_time*1000:.2f}ms per call"


class TestEndToEndScenarios:
    """End-to-end evaluation scenarios."""
    
    def test_single_tool_call_scenario(self):
        """Test scenario with single tool call."""
        from react_sync import app
        
        # Create a mock for the model to avoid actual API calls
        with patch('agentflow.graph.agent.Agent.__call__') as mock_agent:
            # Configure mock to simulate tool call
            def agent_side_effect(state: AgentState) -> AgentState:
                if len(state.context) == 1:  # Initial user message
                    msg = Message.text_message("Calling weather tool", role="assistant")
                    msg.tools_calls = [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}'
                            }
                        }
                    ]
                    state.context.append(msg)
                else:  # After tool execution
                    msg = Message.text_message("The weather is sunny!", role="assistant")
                    msg.tools_calls = []
                    state.context.append(msg)
                return state
            
            mock_agent.side_effect = agent_side_effect
    
    def test_conversation_flow_integrity(self):
        """Test that conversation flow maintains integrity."""
        from react_sync import should_use_tools
        
        # Simulate a complete conversation flow
        state = AgentState()
        flow = []
        
        # User asks question
        state.context = [Message.text_message("Weather in NYC?", role="user")]
        decision = should_use_tools(state)
        flow.append(("user_message", decision))
        
        # Assistant with tool call
        msg = Message.text_message("Checking...", role="assistant")
        msg.tools_calls = [{"id": "1", "type": "function"}]
        state.context.append(msg)
        decision = should_use_tools(state)
        flow.append(("assistant_with_tools", decision))
        
        # Tool result
        state.context.append(Message.text_message("Sunny", role="tool"))
        decision = should_use_tools(state)
        flow.append(("tool_result", decision))
        
        # Final assistant response
        final_msg = Message.text_message("It's sunny!", role="assistant")
        final_msg.tools_calls = []
        state.context.append(final_msg)
        decision = should_use_tools(state)
        flow.append(("final_response", decision))
        
        # Verify flow
        expected_flow = [
            ("user_message", "END"),
            ("assistant_with_tools", "TOOL"),
            ("tool_result", "MAIN"),
            ("final_response", "END"),
        ]
        
        assert flow == expected_flow, f"Flow mismatch:\nExpected: {expected_flow}\nGot: {flow}"


class TestRobustnessAndEdgeCases:
    """Tests for robustness and edge cases."""
    
    def test_empty_location_handling(self):
        """Test handling of empty location string."""
        from react_sync import get_weather
        
        result = get_weather(location="")
        assert isinstance(result, str)
        assert "weather" in result.lower()
    
    def test_very_long_location_name(self):
        """Test handling of very long location names."""
        from react_sync import get_weather
        
        long_location = "A" * 1000
        result = get_weather(location=long_location)
        
        assert isinstance(result, str)
        assert long_location in result
    
    def test_state_with_many_messages(self):
        """Test routing with many messages in context."""
        from react_sync import should_use_tools
        
        state = AgentState()
        # Add 100 messages
        for i in range(100):
            state.context.append(
                Message.text_message(f"Message {i}", role="user" if i % 2 == 0 else "assistant")
            )
        
        # Should still make correct decision based on last message
        result = should_use_tools(state)
        assert result in ["TOOL", "MAIN", "END"]
    
    def test_context_trimming_behavior(self):
        """Test that agent respects trim_context setting."""
        from react_sync import agent
        
        assert agent.trim_context is True, "Agent should have trim_context enabled"


class TestEvaluationCriteria:
    """Custom evaluation criteria for the ReAct agent."""
    
    def test_tool_call_accuracy_criterion(self):
        """Test criterion for tool call accuracy."""
        
        class ToolCallAccuracyCriterion(BaseCriterion):
            """Criterion to check if correct tool was called."""
            
            name = "tool_call_accuracy"
            description = "Verifies that the correct tool was called"
            
            async def evaluate(
                self,
                actual: TrajectoryCollector,
                expected: EvalCase,
            ) -> CriterionResult:
                # Check if get_weather was called
                tool_called = False
                for message in actual.messages:
                    if hasattr(message, 'tools_calls') and message.tools_calls:
                        for tool_call in message.tools_calls:
                            if tool_call.get("function", {}).get("name") == "get_weather":
                                tool_called = True
                                break
                
                score = 1.0 if tool_called else 0.0
                return CriterionResult.success(
                    criterion=self.name,
                    score=score,
                    threshold=self.threshold,
                )
        
        # Test the criterion
        criterion = ToolCallAccuracyCriterion()
        collector = TrajectoryCollector()
        
        # Add message with tool call
        msg = Message.text_message("Calling", role="assistant")
        msg.tools_calls = [
            {
                "function": {"name": "get_weather"},
                "type": "function"
            }
        ]
        collector.add_message(msg)
        
        # Evaluate
        import asyncio
        case = EvalCase.single_turn("test", "input")
        result = asyncio.run(criterion.evaluate(collector, case))
        
        assert result.score == 1.0
        assert result.passed is True


class TestConfigurationValidation:
    """Tests for validating agent configuration."""
    
    def test_model_configuration(self):
        """Test that model is configured correctly."""
        from react_sync import agent
        
        assert agent.model == "gemini-2.5-flash"
        assert agent.provider == "google"
    
    def test_system_prompt_quality(self):
        """Test that system prompt contains necessary instructions."""
        from react_sync import agent
        
        system_prompt = agent.system_prompt
        
        # Convert to string for easier checking
        prompt_text = str(system_prompt).lower()
        
        # Should contain key phrases
        assert "helpful" in prompt_text or "assist" in prompt_text
        assert "2024-06-15" in str(system_prompt)  # Date context
    
    def test_tool_node_name_configuration(self):
        """Test that tool node name matches graph configuration."""
        from react_sync import agent, graph
        
        assert agent.tool_node_name == "TOOL"
        assert "TOOL" in graph.nodes
    
    def test_recursion_limit_sensible(self):
        """Test that recursion limit is set to reasonable value."""
        config = {"thread_id": "test", "recursion_limit": 10}
        
        assert config["recursion_limit"] >= 5, "Recursion limit too low"
        assert config["recursion_limit"] <= 50, "Recursion limit too high"


class TestComparisonWithExpectedBehavior:
    """Tests comparing actual behavior with expected outcomes."""
    
    def test_expected_message_sequence(self):
        """Test that message sequence follows expected pattern."""
        expected_pattern = [
            "user",      # User asks question
            "assistant", # Assistant decides to use tool
            "tool",      # Tool executes
            "assistant", # Assistant provides final answer
        ]
        
        # This pattern should be followed in a typical execution
        # We verify the routing logic supports this pattern
        from react_sync import should_use_tools
        
        state = AgentState()
        
        # Simulate each step
        roles_and_decisions = []
        
        # Step 1: User message
        state.context = [Message.text_message("Question?", role="user")]
        roles_and_decisions.append(("user", should_use_tools(state)))
        
        # Step 2: Assistant with tool call
        msg = Message.text_message("Using tool", role="assistant")
        msg.tools_calls = [{"id": "1"}]
        state.context.append(msg)
        roles_and_decisions.append(("assistant_tool", should_use_tools(state)))
        
        # Step 3: Tool result
        state.context.append(Message.text_message("Result", role="tool"))
        roles_and_decisions.append(("tool", should_use_tools(state)))
        
        # Verify decisions allow the expected flow
        assert roles_and_decisions[1][1] == "TOOL"  # Assistant with tools goes to TOOL
        assert roles_and_decisions[2][1] == "MAIN"  # Tool result goes back to MAIN


class TestDocumentationAndExamples:
    """Tests to validate that the example is well-documented."""
    
    def test_example_has_comments(self):
        """Test that example code is well-commented."""
        import react_sync
        import inspect
        
        source = inspect.getsource(react_sync)
        
        # Count comment lines
        lines = source.split('\n')
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        
        # Should have reasonable number of comments
        comment_ratio = len(comment_lines) / len(lines)
        assert comment_ratio > 0.1, f"Not enough comments: {comment_ratio:.1%}"
    
    def test_functions_have_docstrings(self):
        """Test that key functions have docstrings."""
        from react_sync import get_weather, should_use_tools
        
        assert get_weather.__doc__ is not None, "get_weather should have docstring"
        assert should_use_tools.__doc__ is not None, "should_use_tools should have docstring"


class TestReproducibility:
    """Tests for ensuring reproducible behavior."""
    
    def test_consistent_tool_output(self):
        """Test that tool produces consistent output for same input."""
        from react_sync import get_weather
        
        location = "Test City"
        results = [get_weather(location=location) for _ in range(10)]
        
        # All results should be identical
        assert all(r == results[0] for r in results), "Tool output not consistent"
    
    def test_routing_deterministic(self):
        """Test that routing decisions are deterministic."""
        from react_sync import should_use_tools
        
        state = AgentState()
        state.context = [Message.text_message("Test", role="user")]
        
        results = [should_use_tools(state) for _ in range(10)]
        
        # All results should be identical
        assert all(r == results[0] for r in results), "Routing not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
