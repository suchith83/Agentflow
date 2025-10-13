"""Comprehensive tests for the React prebuilt agent."""

import pytest
from unittest.mock import Mock, patch

from agentflow.checkpointer import InMemoryCheckpointer
from agentflow.graph import ToolNode, CompiledGraph
from agentflow.prebuilt.agent.react import ReactAgent, _should_use_tools
from agentflow.state import AgentState, Message
from agentflow.utils import END
from agentflow.utils.callbacks import CallbackManager


class TestReactAgent:
    """Test the ReactAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state = AgentState()
        self.react_agent = ReactAgent[AgentState](state=self.state)
        
    def test_init_default(self):
        """Test ReactAgent initialization with defaults."""
        agent = ReactAgent[AgentState]()
        assert agent is not None
        assert agent._graph is not None
        
    def test_init_with_state(self):
        """Test ReactAgent initialization with custom state."""
        state = AgentState()
        agent = ReactAgent[AgentState](state=state)
        assert agent is not None
        assert agent._graph is not None
        
    def test_compile_with_callable_nodes(self):
        """Test compiling ReactAgent with callable main and tool nodes."""
        def mock_main_node(state: AgentState) -> AgentState:
            state.context.append(Message.text_message("Main node executed"))
            return state
            
        def mock_tool_node(state: AgentState) -> AgentState:
            """Mock tool node function that handles tool calls."""
            # Simulate tool execution
            if state.context and hasattr(state.context[-1], 'tools_calls'):
                tool_result = Message.text_message("Tool executed successfully", role="tool")
                state.context.append(tool_result)
            return state
        
        compiled = self.react_agent.compile(
            main_node=mock_main_node,
            tool_node=mock_tool_node,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_tuple_nodes(self):
        """Test compiling ReactAgent with tuple (function, name) nodes."""
        def mock_main_node(state: AgentState) -> AgentState:
            state.context.append(Message.text_message("Custom main executed"))
            return state
            
        def mock_tool_node(state: AgentState) -> AgentState:
            """Custom tool node function."""
            tool_result = Message.text_message("Custom tool executed", role="tool")
            state.context.append(tool_result)
            return state
        
        compiled = self.react_agent.compile(
            main_node=(mock_main_node, "CUSTOM_MAIN"),
            tool_node=(mock_tool_node, "CUSTOM_TOOL"),
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_checkpointer(self):
        """Test compiling ReactAgent with checkpointer."""
        def mock_main_node(state: AgentState) -> AgentState:
            return state
            
        def mock_tool_node(state: AgentState) -> AgentState:
            return state
            
        checkpointer = InMemoryCheckpointer[AgentState]()
        
        compiled = self.react_agent.compile(
            main_node=mock_main_node,
            tool_node=mock_tool_node,
            checkpointer=checkpointer,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_callback_manager(self):
        """Test compiling ReactAgent with callback manager."""
        def mock_main_node(state: AgentState) -> AgentState:
            return state
            
        def mock_tool_node(state: AgentState) -> AgentState:
            return state
            
        callback_manager = CallbackManager()
        
        compiled = self.react_agent.compile(
            main_node=mock_main_node,
            tool_node=mock_tool_node,
            callback_manager=callback_manager,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_interrupts(self):
        """Test compiling ReactAgent with interrupt configurations."""
        def mock_main_node(state: AgentState) -> AgentState:
            return state
            
        def mock_tool_node(state: AgentState) -> AgentState:
            return state
            
        compiled = self.react_agent.compile(
            main_node=mock_main_node,
            tool_node=mock_tool_node,
            interrupt_before=["MAIN"],
            interrupt_after=["TOOL"],
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_invalid_main_node_tuple(self):
        """Test error handling for invalid main node in tuple format."""
        def mock_tool_node(state: AgentState) -> AgentState:
            return state
        
        with pytest.raises(ValueError, match="main_node\\[0\\] must be a callable function"):
            self.react_agent.compile(
                main_node=("not_callable", "MAIN"),
                tool_node=mock_tool_node,
            )
            
    def test_compile_invalid_main_node_callable(self):
        """Test error handling for invalid main node as direct value."""
        def mock_tool_node(state: AgentState) -> AgentState:
            return state
            
        with pytest.raises(ValueError, match="main_node must be a callable function"):
            self.react_agent.compile(
                main_node="not_callable",
                tool_node=mock_tool_node,
            )
            
    def test_compile_invalid_tool_node_tuple(self):
        """Test error handling for invalid tool node in tuple format."""
        def mock_main_node(state: AgentState) -> AgentState:
            return state
            
        with pytest.raises(ValueError, match="tool_node\\[0\\] must be a callable function"):
            self.react_agent.compile(
                main_node=mock_main_node,
                tool_node=("not_callable", "TOOL"),
            )
            
    def test_compile_invalid_tool_node_callable(self):
        """Test error handling for invalid tool node as direct value."""
        def mock_main_node(state: AgentState) -> AgentState:
            return state
            
        with pytest.raises(ValueError, match="tool_node must be a callable function"):
            self.react_agent.compile(
                main_node=mock_main_node,
                tool_node="not_callable",
            )


class TestShouldUseToolsFunction:
    """Test the _should_use_tools conditional function."""
    
    def test_empty_context(self):
        """Test with empty context."""
        state = AgentState()
        result = _should_use_tools(state)
        assert result == "TOOL"
        
    def test_no_context(self):
        """Test with None context."""
        state = AgentState()
        state.context = []  # Empty list instead of None
        result = _should_use_tools(state)
        assert result == "TOOL"
        
    def test_assistant_with_tool_calls(self):
        """Test with assistant message that has tool calls."""
        state = AgentState()
        
        # Create a mock message with tool calls
        message = Message.text_message("I need to call a tool", role="assistant")
        message.tools_calls = [{"id": "call_123", "type": "function", "function": {"name": "test_tool"}}]
        
        state.context = [message]
        result = _should_use_tools(state)
        assert result == "TOOL"
        
    def test_assistant_without_tool_calls(self):
        """Test with assistant message without tool calls."""
        state = AgentState()
        
        message = Message.text_message("Just a regular response", role="assistant")
        message.tools_calls = []
        
        state.context = [message]
        result = _should_use_tools(state)
        assert result == END
        
    def test_tool_result_message(self):
        """Test with tool result message."""
        state = AgentState()
        
        message = Message.text_message("Tool executed successfully", role="tool")
        
        state.context = [message]
        result = _should_use_tools(state)
        assert result == "MAIN"
        
    def test_user_message(self):
        """Test with user message."""
        state = AgentState()
        
        message = Message.text_message("Hello, how are you?", role="user")
        
        state.context = [message]
        result = _should_use_tools(state)
        assert result == END
        
    def test_system_message(self):
        """Test with system message."""
        state = AgentState()
        
        message = Message.text_message("You are a helpful assistant", role="system")
        
        state.context = [message]
        result = _should_use_tools(state)
        assert result == END
        
    def test_mixed_messages_last_is_tool(self):
        """Test with multiple messages where last is tool result."""
        state = AgentState()
        
        user_msg = Message.text_message("Call a tool", role="user")
        
        assistant_msg = Message.text_message("I'll call the tool", role="assistant")
        assistant_msg.tools_calls = [{"id": "call_123", "type": "function"}]
        
        tool_msg = Message.text_message("Tool result", role="tool")
        
        state.context = [user_msg, assistant_msg, tool_msg]
        result = _should_use_tools(state)
        assert result == "MAIN"
        
    def test_mixed_messages_last_is_assistant_with_tools(self):
        """Test with multiple messages where last is assistant with tool calls."""
        state = AgentState()
        
        user_msg = Message.text_message("Call a tool", role="user")
        
        assistant_msg = Message.text_message("I'll call the tool", role="assistant")
        assistant_msg.tools_calls = [{"id": "call_123", "type": "function"}]
        
        state.context = [user_msg, assistant_msg]
        result = _should_use_tools(state)
        assert result == "TOOL"


class TestReactAgentIntegration:
    """Integration tests for the ReactAgent."""
    
    def test_react_agent_execution_flow(self):
        """Test the complete React agent execution flow."""
        def mock_main_node(state: AgentState) -> AgentState:
            # Simulate AI making tool calls
            if len(state.context) == 1:  # First call, user message
                msg = Message.text_message("I need to call a tool", role="assistant")
                msg.tools_calls = [{"id": "call_123", "type": "function", "function": {"name": "test_tool", "arguments": '{"input": "test"}'}}]
                state.context.append(msg)
            else:  # After tool execution, make final response
                msg = Message.text_message("Tool executed, here's the result", role="assistant")
                msg.tools_calls = []
                state.context.append(msg)
            return state
            
        def mock_tool_function(input: str) -> str:
            return f"Tool processed: {input}"
            
        tool_node = ToolNode([mock_tool_function])
        react_agent = ReactAgent[AgentState]()

        compiled = react_agent.compile(
            main_node=mock_main_node,
            tool_node=tool_node,
        )        # Execute the agent
        initial_state = {"messages": [Message.text_message("Please use the tool", role="user")]}
        result = compiled.invoke(initial_state, config={"thread_id": "test_123"})
        
        assert isinstance(result, dict)
        assert "messages" in result
        
    @pytest.mark.asyncio
    async def test_react_agent_async_execution(self):
        """Test React agent with async execution."""
        async def mock_async_main_node(state: AgentState) -> AgentState:
            msg = Message.text_message("Async response", role="assistant")
            state.context.append(msg)
            return state
            
        def mock_tool_node(state: AgentState) -> AgentState:
            tool_result = Message.text_message("Async tool executed", role="tool")
            state.context.append(tool_result)
            return state
        react_agent = ReactAgent[AgentState]()
        
        compiled = react_agent.compile(
            main_node=mock_async_main_node,
            tool_node=mock_tool_node,
        )
        
        # Execute the agent asynchronously
        initial_state = {"messages": [Message.text_message("Test async", role="user")]}
        result = await compiled.ainvoke(initial_state, config={"thread_id": "async_test"})
        
        assert isinstance(result, dict)
        assert "messages" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])