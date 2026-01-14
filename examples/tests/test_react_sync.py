"""
Unit tests for the react_sync.py example.

This module tests individual components and functions of the ReAct agent example,
including the weather tool, routing logic, and graph construction.
"""

# Import the components from react_sync
import sys
from pathlib import Path

import pytest

from agentflow.checkpointer import InMemoryCheckpointer
from agentflow.graph import ToolNode
from agentflow.graph.agent import Agent
from agentflow.state import AgentState, Message
from agentflow.utils.constants import END


# Add the react directory to the path so we can import from react_sync
react_dir = Path(__file__).parent
sys.path.insert(0, str(react_dir))


class TestGetWeatherTool:
    """Tests for the get_weather tool function."""

    def test_get_weather_basic_call(self):
        """Test basic weather tool call without injected parameters."""
        from react_sync import get_weather

        result = get_weather(location="New York City")

        assert result == "The weather in New York City is sunny"
        assert isinstance(result, str)

    def test_get_weather_with_tool_call_id(self, capsys):
        """Test weather tool with injected tool_call_id."""
        from react_sync import get_weather

        result = get_weather(location="London", tool_call_id="call_12345")

        captured = capsys.readouterr()
        assert "Tool call ID: call_12345" in captured.out
        assert result == "The weather in London is sunny"

    def test_get_weather_with_state(self, capsys):
        """Test weather tool with injected state."""
        from react_sync import get_weather

        state = AgentState()
        state.context = [
            Message.text_message("Hello", role="user"),
            Message.text_message("Hi there", role="assistant"),
        ]

        result = get_weather(location="Paris", tool_call_id="call_123", state=state)

        captured = capsys.readouterr()
        assert "Number of messages in context: 2" in captured.out
        assert result == "The weather in Paris is sunny"

    def test_get_weather_various_locations(self):
        """Test weather tool with various location inputs."""
        from react_sync import get_weather

        locations = ["Tokyo", "Berlin", "Sydney", "Toronto"]

        for location in locations:
            result = get_weather(location=location)
            assert f"The weather in {location} is sunny" == result


class TestShouldUseToolsFunction:
    """Tests for the should_use_tools routing function."""

    def test_empty_context(self):
        """Test routing with empty context."""
        from react_sync import should_use_tools

        state = AgentState()
        state.context = []

        result = should_use_tools(state)
        assert result == "TOOL"

    def test_no_context_attribute(self):
        """Test routing when context is None."""
        from react_sync import should_use_tools

        state = AgentState()
        state.context = None  # type: ignore

        result = should_use_tools(state)
        assert result == "TOOL"

    def test_assistant_with_tool_calls(self):
        """Test routing when assistant message has tool calls."""
        from react_sync import should_use_tools

        state = AgentState()
        message = Message.text_message("I'll call the tool", role="assistant")
        message.tools_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
            }
        ]
        state.context = [message]

        result = should_use_tools(state)
        assert result == "TOOL"

    def test_assistant_without_tool_calls(self):
        """Test routing when assistant message has no tool calls."""
        from react_sync import should_use_tools

        state = AgentState()
        message = Message.text_message("Here's my response", role="assistant")
        message.tools_calls = []
        state.context = [message]

        result = should_use_tools(state)
        assert result == END

    def test_tool_result_message(self):
        """Test routing when last message is a tool result."""
        from react_sync import should_use_tools

        state = AgentState()
        tool_message = Message.text_message("The weather in NYC is sunny", role="tool")
        state.context = [tool_message]

        result = should_use_tools(state)
        assert result == "MAIN"

    def test_user_message(self):
        """Test routing when last message is from user."""
        from react_sync import should_use_tools

        state = AgentState()
        user_message = Message.text_message("What's the weather?", role="user")
        state.context = [user_message]

        result = should_use_tools(state)
        assert result == END

    def test_complex_conversation_flow(self):
        """Test routing in a multi-turn conversation."""
        from react_sync import should_use_tools

        state = AgentState()

        # Step 1: User asks a question
        state.context = [Message.text_message("What's the weather in NYC?", role="user")]
        assert should_use_tools(state) == END

        # Step 2: Assistant decides to call a tool
        assistant_msg = Message.text_message("Let me check", role="assistant")
        assistant_msg.tools_calls = [{"id": "call_1", "type": "function"}]
        state.context.append(assistant_msg)
        assert should_use_tools(state) == "TOOL"

        # Step 3: Tool returns result
        state.context.append(Message.text_message("Sunny", role="tool"))
        assert should_use_tools(state) == "MAIN"

        # Step 4: Assistant makes final response
        final_msg = Message.text_message("It's sunny!", role="assistant")
        final_msg.tools_calls = []
        state.context.append(final_msg)
        assert should_use_tools(state) == END


class TestToolNode:
    """Tests for the ToolNode configuration."""

    def test_tool_node_creation(self):
        """Test that ToolNode is created correctly with get_weather."""
        from react_sync import tool_node, get_weather

        assert isinstance(tool_node, ToolNode)
        # Verify the tool is in the tool node's tools
        tools = tool_node.all_tools_sync()
        assert len(tools) > 0

    @pytest.mark.asyncio
    async def test_tool_node_execution(self):
        """Test executing the tool node with a mock state."""
        from react_sync import tool_node

        state = AgentState()
        assistant_msg = Message.text_message("Calling tool", role="assistant")
        assistant_msg.tools_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Boston"}'},
            }
        ]
        state.context = [assistant_msg]

        # Execute the tool node
        result_state = await tool_node.invoke(
            name="get_weather",
            args={"location": "Boston"},
            tool_call_id="call_123",
            config={"user_id": "test_user"},
            state=state,
        )

        # print("result_state:", result_state)

        # Verify tool result message was returned
        assert isinstance(result_state, Message)
        # The returned message should be a tool result
        assert getattr(result_state, "type", None) == "tool_result"
class TestAgentConfiguration:
    """Tests for the Agent configuration."""

    def test_agent_creation(self):
        """Test that agent is created with correct configuration."""
        from react_sync import agent

        assert isinstance(agent, Agent)
        assert agent.model == "gemini-2.5-flash"
        assert agent.provider == "google"
        assert agent.tool_node_name == "TOOL"
        assert agent.trim_context is True

    def test_agent_system_prompt(self):
        """Test that agent has correct system prompt."""
        from react_sync import agent

        assert agent.system_prompt is not None
        assert len(agent.system_prompt) == 2

        # Check first system message
        assert agent.system_prompt[0]["role"] == "system"
        assert "helpful assistant" in agent.system_prompt[0]["content"]

        # Check date context
        assert agent.system_prompt[1]["role"] == "user"
        assert "2024-06-15" in agent.system_prompt[1]["content"]


class TestGraphConstruction:
    """Tests for the StateGraph construction and compilation."""

    def test_graph_nodes(self):
        """Test that graph has correct nodes."""
        from react_sync import graph

        # Graph should have MAIN and TOOL nodes
        assert "MAIN" in graph.nodes
        assert "TOOL" in graph.nodes

    def test_graph_edges(self):
        """Test that graph has correct edges."""
        from react_sync import graph

        # TOOL should have edge to MAIN
        tool_to_main_found = False
        for edge in graph.edges:
            if edge.from_node == "TOOL" and edge.to_node == "MAIN":
                tool_to_main_found = True
                break
        assert tool_to_main_found, "Edge from TOOL to MAIN not found"


    def test_graph_conditional_edges(self):
        """Test that graph has correct conditional edges."""
        from react_sync import graph

        # MAIN should have conditional edges
        conditional_edges = [edge for edge in graph.edges if edge.from_node == "MAIN" and edge.condition is not None]
        assert len(conditional_edges) > 0
        conditional_edge = conditional_edges[0]

        # Check that it routes to TOOL or END
        assert callable(conditional_edge.condition)
        target_nodes = {edge.to_node for edge in conditional_edges}
        assert "TOOL" in target_nodes
        assert END in target_nodes
        # assert END in conditional_edge.path_map.values()

    def test_graph_entry_point(self):
        """Test that graph has correct entry point."""
        from react_sync import graph

        assert graph.entry_point == "MAIN"

    def test_graph_compilation(self):
        """Test that graph compiles successfully."""
        from react_sync import app

        assert app is not None
        # Verify it's a compiled graph
        assert hasattr(app, "invoke")
        assert hasattr(app, "ainvoke")
        assert hasattr(app, "stream")


class TestCheckpointerConfiguration:
    """Tests for checkpointer configuration."""

    def test_checkpointer_type(self):
        """Test that checkpointer is InMemoryCheckpointer."""
        from react_sync import checkpointer

        assert isinstance(checkpointer, InMemoryCheckpointer)

    def test_app_has_checkpointer(self):
        """Test that compiled app uses the configured checkpointer instance.

        This test intentionally touches the private ``_checkpointer`` attribute on the
        compiled graph object to verify that the ``react_sync.checkpointer`` is the
        same instance wired into ``app``. If a public API for accessing the
        checkpointer is added in the future, this test should be updated to use it
        instead of relying on the private attribute.
        """
        from react_sync import app, checkpointer

        # NOTE: This assertion intentionally verifies an internal attribute to ensure
        # that the compiled graph is wired to the expected checkpointer instance.
        assert app._checkpointer == checkpointer


class TestIntegration:
    """Integration tests for the complete ReAct agent setup."""

    @pytest.fixture(autouse=True)
    def skip_integration(self, request):
        try:
            run_integration = request.config.getoption("--run-integration")
        except ValueError:
            run_integration = False

        if not run_integration:
            pytest.skip("Integration tests require --run-integration flag")

    def test_full_agent_execution(self):
        """Test complete agent execution flow."""
        from react_sync import app

        inp = {"messages": [Message.text_message("Please call the get_weather function for Tokyo")]}

        # This would require actual API calls, so we skip in normal tests
        # config = {"thread_id": "test_12345", "recursion_limit": 10}
        # result = app.invoke(inp, config=config)
        #
        # assert "messages" in result
        # assert len(result["messages"]) > 1

    def test_graph_structure_integrity(self):
        """Test that the graph structure is valid."""
        from react_sync import graph

        # Entry point should be valid
        assert graph.entry_point in graph.nodes

        # All edges should connect valid nodes or END
        for edge in graph.edges:
            assert edge.from_node in graph.nodes
            assert edge.to_node in graph.nodes or edge.to_node == END

        # Conditional edges should have valid sources
        for source in graph.conditional_edges.keys():
            assert source in graph.nodes


class TestMessageFlow:
    """Tests for message flow through the agent."""

    def test_initial_message_format(self):
        """Test that initial messages are formatted correctly."""
        inp = {"messages": [Message.text_message("Test message")]}

        assert "messages" in inp
        assert isinstance(inp["messages"], list)
        assert len(inp["messages"]) == 1
        assert inp["messages"][0].role == "user"

    def test_config_structure(self):
        """Test that config is structured correctly."""
        config = {"thread_id": "12345", "recursion_limit": 10}

        assert "thread_id" in config
        assert "recursion_limit" in config
        assert isinstance(config["thread_id"], str)
        assert isinstance(config["recursion_limit"], int)


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_empty_state_handling(self):
        """Test that empty state is handled gracefully."""
        from react_sync import should_use_tools

        state = AgentState()
        state.context = []

        # Should not raise an exception
        result = should_use_tools(state)
        assert result == "TOOL"

    def test_invalid_tool_call_structure(self):
        """Test handling of messages without proper tools_calls attribute."""
        from react_sync import should_use_tools

        state = AgentState()
        message = Message.text_message("Test", role="assistant")
        # Don't set tools_calls attribute
        state.context = [message]

        # Should handle gracefully and not crash
        result = should_use_tools(state)
        # Should route to END since no tool calls are present
        assert result == END


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
