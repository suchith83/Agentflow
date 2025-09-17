# """Comprehensive multiagent test suite for PyAgenity."""

# import asyncio

# import pytest
# from injectq import Inject, InjectQ, inject
# from pydantic import Field

# from pyagenity.exceptions import NodeError
# from pyagenity.graph import CompiledGraph, StateGraph
# from pyagenity.publisher import ConsolePublisher
# from pyagenity.state import AgentState
# from pyagenity.utils import END, Message


# class TestMultiAgentSuite:
#     """Comprehensive test suite for multiagent scenarios."""

#     @pytest.fixture
#     def container(self):
#         """InjectQ container for dependency injection tests."""
#         container = InjectQ.get_instance()
#         # Clear existing bindings by creating a new instance
#         return container

#     @pytest.fixture
#     def agent_state(self):
#         """Basic agent state fixture."""
#         return AgentState()

#     def create_simple_agent_graph(self) -> CompiledGraph:
#         """Create a simple agent graph for testing."""

#         def simple_agent(state: AgentState) -> Message:
#             # Return a message that will be added to the state
#             return Message.from_text("Agent processed message", "assistant")

#         graph = StateGraph()
#         graph.add_node("agent", simple_agent)
#         graph.set_entry_point("agent")
#         graph.add_edge("agent", END)

#         return graph.compile()

#     def test_single_agent_invoke_sync(self):
#         """Test single agent with synchronous invoke."""
#         graph = self.create_simple_agent_graph()
#         messages = [Message.from_text("Hello from user", "user")]

#         result = graph.invoke({"messages": messages})

#         assert "messages" in result
#         assert len(result["messages"]) == 2  # input + output message
#         assert result["messages"][-1].content == "Agent processed message"

#     @pytest.mark.asyncio
#     async def test_single_agent_invoke_async(self):
#         """Test single agent with asynchronous invoke."""
#         graph = self.create_simple_agent_graph()
#         messages = [Message.from_text("Hello from user", "user")]

#         result = await graph.ainvoke({"messages": messages})

#         assert "messages" in result
#         assert len(result["messages"]) == 2  # input + output message
#         assert result["messages"][-1].content == "Agent processed message"

#     def test_single_agent_stream_sync(self):
#         """Test single agent with synchronous streaming."""
#         graph = self.create_simple_agent_graph()
#         messages = [Message.from_text("Hello from user", "user")]

#         events = list(graph.stream({"messages": messages}))

#         assert len(events) >= 1
#         # Check that we have events with the expected content
#         found_processed = False
#         for event in events:
#             if hasattr(event, "content") and "Agent processed message" in event.content:
#                 found_processed = True
#                 break
#         assert found_processed

#     @pytest.mark.asyncio
#     async def test_single_agent_stream_async(self):
#         """Test single agent with asynchronous streaming."""
#         graph = self.create_simple_agent_graph()
#         messages = [Message.from_text("Hello from user", "user")]

#         # Fix: astream returns a coroutine that yields events, need to await it first
#         stream_coro = graph.astream({"messages": messages})
#         events = []
#         async for event in stream_coro:
#             events.append(event)

#         assert len(events) >= 1
#         found_processed = False
#         for event in events:
#             if hasattr(event, "content") and "Agent processed message" in event.content:
#                 found_processed = True
#                 break
#         assert found_processed

#     def test_five_agent_sequential_workflow_sync(self):
#         """Test 5 agents in sequential workflow with sync invoke."""

#         def create_agent_node(agent_id: str):
#             def agent_func(state: AgentState) -> Message:
#                 return Message.from_text(f"Agent {agent_id} processed", "assistant")

#             return agent_func

#         graph = StateGraph()

#         # Add 5 agents
#         for i in range(1, 6):
#             graph.add_node(f"agent_{i}", create_agent_node(str(i)))

#         # Set up sequential edges
#         graph.set_entry_point("agent_1")
#         for i in range(1, 5):
#             graph.add_edge(f"agent_{i}", f"agent_{i + 1}")
#         graph.add_edge("agent_5", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Start workflow", "user")]
#         result = compiled.invoke({"messages": messages})

#         assert "messages" in result
#         assert len(result["messages"]) >= 6  # input + 5 agent messages

#         # Check that all agents contributed
#         content = " ".join([msg.content for msg in result["messages"]])
#         for i in range(1, 6):
#             assert f"Agent {i} processed" in content

#     @pytest.mark.asyncio
#     async def test_five_agent_sequential_workflow_async(self):
#         """Test 5 agents in sequential workflow with async invoke."""

#         def create_agent_node(agent_id: str):
#             def agent_func(state: AgentState) -> Message:
#                 return Message.from_text(f"Agent {agent_id} processed", "assistant")

#             return agent_func

#         graph = StateGraph()

#         # Add 5 agents
#         for i in range(1, 6):
#             graph.add_node(f"agent_{i}", create_agent_node(str(i)))

#         # Set up sequential edges
#         graph.set_entry_point("agent_1")
#         for i in range(1, 5):
#             graph.add_edge(f"agent_{i}", f"agent_{i + 1}")
#         graph.add_edge("agent_5", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Start workflow", "user")]
#         result = await compiled.ainvoke({"messages": messages})

#         assert "messages" in result
#         assert len(result["messages"]) >= 6  # input + 5 agent messages

#         # Check that all agents contributed
#         content = " ".join([msg.content for msg in result["messages"]])
#         for i in range(1, 6):
#             assert f"Agent {i} processed" in content

#     def test_ten_agent_parallel_workflow_sync(self):
#         """Test 10 agents in parallel workflow with sync invoke."""

#         def create_agent_node(agent_id: str):
#             def agent_func(state: AgentState) -> Message:
#                 return Message.from_text(f"Agent {agent_id} processed", "assistant")

#             return agent_func

#         graph = StateGraph()

#         # Add 10 parallel agents
#         for i in range(1, 11):
#             graph.add_node(f"agent_{i}", create_agent_node(str(i)))

#         # Set up parallel execution - all agents run independently
#         graph.set_entry_point("agent_1")
#         for i in range(1, 11):
#             graph.add_edge(f"agent_{i}", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Parallel processing", "user")]
#         result = compiled.invoke({"messages": messages})

#         assert "messages" in result
#         # Should have input + at least one agent message
#         assert len(result["messages"]) >= 2

#     @pytest.mark.asyncio
#     async def test_ten_agent_parallel_workflow_async(self):
#         """Test 10 agents in parallel workflow with async invoke."""

#         def create_agent_node(agent_id: str):
#             def agent_func(state: AgentState) -> Message:
#                 return Message.from_text(f"Agent {agent_id} processed", "assistant")

#             return agent_func

#         graph = StateGraph()

#         # Add 10 parallel agents
#         for i in range(1, 11):
#             graph.add_node(f"agent_{i}", create_agent_node(str(i)))

#         # Set up parallel execution - all agents run independently
#         graph.set_entry_point("agent_1")
#         for i in range(1, 11):
#             graph.add_edge(f"agent_{i}", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Parallel processing", "user")]
#         result = await compiled.ainvoke({"messages": messages})

#         assert "messages" in result
#         # Should have input + at least one agent message
#         assert len(result["messages"]) >= 2

#     def test_multiagent_with_dependency_injection(self, container):
#         """Test multiagent scenario with InjectQ dependency injection."""
#         # Set up dependencies
#         container[str] = "injected_message"
#         container[int] = 42

#         @inject
#         def agent_with_injection(
#             state: AgentState, message: str = Inject[str], number: int = Inject[int]
#         ) -> Message:
#             return Message.from_text(
#                 f"Agent processed with injection: {message} - {number}", "assistant"
#             )

#         graph = StateGraph(container=container)
#         graph.add_node("injected_agent", agent_with_injection)
#         graph.set_entry_point("injected_agent")
#         graph.add_edge("injected_agent", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Test injection", "user")]
#         result = compiled.invoke({"messages": messages})

#         assert "messages" in result
#         assert len(result["messages"]) >= 2
#         content = result["messages"][-1].content
#         assert "injected_message" in content
#         assert "42" in content

#     def test_sync_node_with_ainvoke_pattern(self):
#         """Test sync node that internally calls ainvoke."""

#         async def async_helper(state: AgentState) -> Message:
#             # Simulate async processing
#             await asyncio.sleep(0.01)
#             return Message.from_text("Async helper processed", "assistant")

#         def sync_node_with_async_call(state: AgentState) -> Message:
#             # This is a sync node that calls async code
#             # In real scenarios, you'd use asyncio.run() or similar
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             try:
#                 result = loop.run_until_complete(async_helper(state))
#                 return result
#             finally:
#                 loop.close()

#         graph = StateGraph()
#         graph.add_node("sync_async_node", sync_node_with_async_call)
#         graph.set_entry_point("sync_async_node")
#         graph.add_edge("sync_async_node", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Test sync-async", "user")]
#         result = compiled.invoke({"messages": messages})

#         assert "messages" in result
#         assert len(result["messages"]) >= 2
#         content = result["messages"][-1].content
#         assert "Async helper processed" in content

#     def test_non_streamable_node_in_stream_context(self):
#         """Test node that doesn't support streaming in a streaming context."""

#         def non_streamable_node(state: AgentState) -> Message:
#             # This node doesn't yield intermediate results
#             return Message.from_text("Non-streamable result", "assistant")

#         graph = StateGraph()
#         graph.add_node("non_streamable", non_streamable_node)
#         graph.set_entry_point("non_streamable")
#         graph.add_edge("non_streamable", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Test non-streamable", "user")]

#         # Test streaming - should still work but may not have intermediate events
#         events = list(compiled.stream({"messages": messages}))

#         # Should have at least one event (final result)
#         assert len(events) >= 1

#         # Check final result
#         result = compiled.invoke({"messages": messages})
#         assert "messages" in result
#         assert len(result["messages"]) >= 2
#         content = result["messages"][-1].content
#         assert "Non-streamable result" in content

#     def test_multiagent_with_tools_and_injection(self, container):
#         """Test multiagent scenario with tools and dependency injection."""
#         # Set up tool dependency
#         container["tool_config"] = {"max_retries": 3, "timeout": 30}

#         def tool_agent(
#             state: AgentState, tool_config: dict | None = Inject["tool_config"]
#         ) -> Message:
#             config_info = f"Tool config: retries={tool_config['max_retries']}"
#             return Message.from_text(f"Tool agent processed with {config_info}", "assistant")

#         def analysis_agent(state: AgentState) -> Message:
#             return Message.from_text("Analysis agent processed", "assistant")

#         graph = StateGraph(container=container)
#         graph.add_node("tool_agent", tool_agent)
#         graph.add_node("analysis_agent", analysis_agent)

#         graph.set_entry_point("tool_agent")
#         graph.add_edge("tool_agent", "analysis_agent")
#         graph.add_edge("analysis_agent", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Multiagent with tools", "user")]
#         result = compiled.invoke({"messages": messages})

#         assert "messages" in result
#         assert len(result["messages"]) >= 3  # input + 2 agent messages
#         content = " ".join([msg.content for msg in result["messages"]])
#         assert "Tool agent processed" in content
#         assert "Analysis agent processed" in content
#         assert "retries=3" in content

#     @pytest.mark.asyncio
#     async def test_multiagent_streaming_workflow_async(self):
#         """Test complex multiagent streaming workflow asynchronously."""

#         # Create a more complex workflow with 7 agents
#         def create_complex_agent(agent_type: str):
#             def agent_func(state: AgentState) -> Message:
#                 message = f"{agent_type.title()} agent processed {len(state.context)} messages"
#                 return Message.from_text(message, "assistant")

#             return agent_func

#         graph = StateGraph()

#         # Add 7 agents
#         agent_types = [
#             "coordinator",
#             "researcher",
#             "analyzer",
#             "validator",
#             "summarizer",
#             "reviewer",
#             "finalizer",
#         ]
#         for agent_type in agent_types:
#             graph.add_node(agent_type, create_complex_agent(agent_type))

#         # Set up workflow edges
#         graph.set_entry_point("coordinator")
#         graph.add_edge("coordinator", "researcher")
#         graph.add_edge("researcher", "analyzer")
#         graph.add_edge("researcher", "validator")
#         graph.add_edge("analyzer", "summarizer")
#         graph.add_edge("validator", "summarizer")
#         graph.add_edge("summarizer", "reviewer")
#         graph.add_edge("reviewer", "finalizer")
#         graph.add_edge("finalizer", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Complex workflow test", "user")]

#         # Test async streaming
#         stream_coro = compiled.astream({"messages": messages})
#         events = []
#         async for event in stream_coro:
#             events.append(event)

#         assert len(events) >= 1

#         # Test async invoke
#         result = await compiled.ainvoke({"messages": messages})
#         assert "messages" in result
#         assert len(result["messages"]) >= 8  # input + 7 agent messages

#         # Verify all agents contributed
#         content = " ".join([msg.content for msg in result["messages"]])
#         for agent_type in agent_types:
#             expected = f"{agent_type.title()} agent processed"
#             assert expected in content

#     def test_error_handling_in_multiagent_workflow(self):
#         """Test error handling across multiple agents."""

#         def failing_agent(state: AgentState) -> Message:
#             raise ValueError("Simulated agent failure")

#         def recovery_agent(state: AgentState) -> Message:
#             return Message.from_text("Recovery agent handled error", "assistant")

#         graph = StateGraph()
#         graph.add_node("failing_agent", failing_agent)
#         graph.add_node("recovery_agent", recovery_agent)

#         graph.set_entry_point("failing_agent")
#         graph.add_edge("failing_agent", "recovery_agent")
#         graph.add_edge("recovery_agent", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Test error handling", "user")]

#         # Test that invoke handles errors gracefully
#         with pytest.raises(ValueError, match="Simulated agent failure"):
#             compiled.invoke({"messages": messages})

#     def test_multiagent_with_custom_state_and_publisher(self):
#         """Test multiagent with custom state and publisher."""
#         publisher = ConsolePublisher()

#         # Create a simple custom state by extending AgentState
#         def metadata_agent(state: AgentState) -> Message:
#             # Store metadata in the state (this would be better with custom state)
#             return Message.from_text("Metadata agent processed", "assistant")

#         def reporting_agent(state: AgentState) -> Message:
#             metadata = getattr(
#                 state,
#                 "agent_metadata",
#                 {"processed_by": "metadata_agent", "message_count": len(state.context)},
#             )
#             report = f"Report: processed by {metadata.get('processed_by')}, {metadata.get('message_count')} messages"
#             return Message.from_text(report, "assistant")

#         graph = StateGraph()
#         graph.add_node("metadata_agent", metadata_agent)
#         graph.add_node("reporting_agent", reporting_agent)

#         graph.set_entry_point("metadata_agent")
#         graph.add_edge("metadata_agent", "reporting_agent")
#         graph.add_edge("reporting_agent", END)

#         compiled = graph.compile()
#         messages = [Message.from_text("Test custom state", "user")]
#         result = compiled.invoke({"messages": messages})

#         assert "messages" in result
#         assert len(result["messages"]) >= 3  # input + 2 agent messages
#         content = " ".join([msg.content for msg in result["messages"]])
#         assert "Metadata agent processed" in content
#         assert "Report:" in content
