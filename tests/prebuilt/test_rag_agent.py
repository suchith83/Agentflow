"""Comprehensive tests for the RAG prebuilt agent."""

import pytest

from pyagenity.checkpointer import InMemoryCheckpointer
from pyagenity.graph import ToolNode, CompiledGraph
from pyagenity.prebuilt.agent.rag import RAGAgent
from pyagenity.state.agent_state import AgentState
from pyagenity.state import Message
from pyagenity.utils import END


class TestRAGAgent:
    """Test the RAGAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state = AgentState()
        self.rag_agent = RAGAgent[AgentState](state=self.state)
        
    def test_init_default(self):
        """Test RAGAgent initialization with defaults."""
        agent = RAGAgent[AgentState]()
        assert agent is not None
        assert agent._graph is not None
        
    def test_init_with_state(self):
        """Test RAGAgent initialization with custom state."""
        state = AgentState()
        agent = RAGAgent[AgentState](state=state)
        assert agent is not None
        assert agent._graph is not None
        
    def test_compile_basic(self):
        """Test basic RAG compilation with retriever and synthesizer."""
        def mock_retriever(state: AgentState) -> AgentState:
            # Simulate document retrieval
            retrieved_msg = Message.text_message("Retrieved document content", role="assistant")
            state.context.append(retrieved_msg)
            return state
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            # Simulate answer synthesis
            answer_msg = Message.text_message("Synthesized answer from retrieved documents", role="assistant")
            state.context.append(answer_msg)
            return state
        
        compiled = self.rag_agent.compile(
            retriever_node=mock_retriever,
            synthesize_node=mock_synthesizer,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_tool_node_retriever(self):
        """Test compilation with ToolNode as retriever."""
        def mock_search_function(query: str) -> str:
            return f"Search results for: {query}"
            
        retriever_tool = ToolNode([mock_search_function])
        
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
        
        compiled = self.rag_agent.compile(
            retriever_node=retriever_tool,
            synthesize_node=mock_synthesizer,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_tuple_nodes(self):
        """Test compilation with tuple node names."""
        def mock_retriever(state: AgentState) -> AgentState:
            return state
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
        
        compiled = self.rag_agent.compile(
            retriever_node=(mock_retriever, "CUSTOM_RETRIEVER"),
            synthesize_node=(mock_synthesizer, "CUSTOM_SYNTHESIZER"),
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_followup_condition(self):
        """Test compilation with followup condition."""
        def mock_retriever(state: AgentState) -> AgentState:
            return state
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
            
        def followup_condition(state: AgentState) -> str:
            # Simple condition: if context has less than 3 messages, retrieve more
            if len(state.context) < 3:
                return "RETRIEVE"
            return END
        
        compiled = self.rag_agent.compile(
            retriever_node=mock_retriever,
            synthesize_node=mock_synthesizer,
            followup_condition=followup_condition,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_checkpointer(self):
        """Test compilation with checkpointer."""
        def mock_retriever(state: AgentState) -> AgentState:
            return state
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
            
        checkpointer = InMemoryCheckpointer[AgentState]()
        
        compiled = self.rag_agent.compile(
            retriever_node=mock_retriever,
            synthesize_node=mock_synthesizer,
            checkpointer=checkpointer,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_with_interrupts(self):
        """Test compilation with interrupt configurations."""
        def mock_retriever(state: AgentState) -> AgentState:
            return state
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
        
        compiled = self.rag_agent.compile(
            retriever_node=mock_retriever,
            synthesize_node=mock_synthesizer,
            interrupt_before=["RETRIEVE"],
            interrupt_after=["SYNTHESIZE"],
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_invalid_retriever_tuple(self):
        """Test error handling for invalid retriever in tuple format."""
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
        
        with pytest.raises(ValueError, match="retriever_node\\[0\\] must be callable or ToolNode"):
            # Type ignore for intentional error testing
            self.rag_agent.compile(
                retriever_node=("not_callable", "RETRIEVER"),  # type: ignore
                synthesize_node=mock_synthesizer,
            )
            
    def test_compile_invalid_retriever_direct(self):
        """Test error handling for invalid retriever as direct value."""
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
            
        with pytest.raises(ValueError, match="retriever_node must be callable or ToolNode"):
            # Type ignore for intentional error testing
            self.rag_agent.compile(
                retriever_node="not_callable",  # type: ignore
                synthesize_node=mock_synthesizer,
            )
            
    def test_compile_invalid_synthesizer_tuple(self):
        """Test error handling for invalid synthesizer in tuple format."""
        def mock_retriever(state: AgentState) -> AgentState:
            return state
            
        with pytest.raises(ValueError, match="synthesize_node\\[0\\] must be callable"):
            # Type ignore for intentional error testing
            self.rag_agent.compile(
                retriever_node=mock_retriever,
                synthesize_node=("not_callable", "SYNTHESIZER"),  # type: ignore
            )
            
    def test_compile_invalid_synthesizer_direct(self):
        """Test error handling for invalid synthesizer as direct value."""
        def mock_retriever(state: AgentState) -> AgentState:
            return state
            
        with pytest.raises(ValueError, match="synthesize_node must be callable"):
            # Type ignore for intentional error testing
            self.rag_agent.compile(
                retriever_node=mock_retriever,
                synthesize_node="not_callable",  # type: ignore
            )


class TestRAGAgentAdvanced:
    """Test the advanced RAG compilation features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state = AgentState()
        self.rag_agent = RAGAgent[AgentState](state=self.state)
        
    def test_compile_advanced_basic(self):
        """Test basic advanced compilation with multiple retrievers."""
        def mock_retriever1(state: AgentState) -> AgentState:
            return state
            
        def mock_retriever2(state: AgentState) -> AgentState:
            return state
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
        
        compiled = self.rag_agent.compile_advanced(
            retriever_nodes=[mock_retriever1, mock_retriever2],
            synthesize_node=mock_synthesizer,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_advanced_with_tool_nodes(self):
        """Test advanced compilation with ToolNode retrievers."""
        def search_func1(query: str) -> str:
            return f"Dense search: {query}"
            
        def search_func2(query: str) -> str:
            return f"Sparse search: {query}"
            
        retriever1 = ToolNode([search_func1])
        retriever2 = ToolNode([search_func2])
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
        
        compiled = self.rag_agent.compile_advanced(
            retriever_nodes=[retriever1, retriever2],
            synthesize_node=mock_synthesizer,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_advanced_with_all_options(self):
        """Test advanced compilation with all optional stages."""
        def mock_query_plan(state: AgentState) -> AgentState:
            return state
            
        def mock_retriever(state: AgentState) -> AgentState:
            return state
            
        def mock_merge(state: AgentState) -> AgentState:
            return state
            
        def mock_rerank(state: AgentState) -> AgentState:
            return state
            
        def mock_compress(state: AgentState) -> AgentState:
            return state
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
            
        def followup_condition(state: AgentState) -> str:
            return END
            
        options = {
            "query_plan": mock_query_plan,
            "merge": mock_merge,
            "rerank": mock_rerank,
            "compress": mock_compress,
            "followup_condition": followup_condition,
        }
        
        compiled = self.rag_agent.compile_advanced(
            retriever_nodes=[mock_retriever],
            synthesize_node=mock_synthesizer,
            options=options,
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_advanced_with_tuple_nodes(self):
        """Test advanced compilation with named nodes via tuples."""
        def mock_retriever(state: AgentState) -> AgentState:
            return state
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
        
        compiled = self.rag_agent.compile_advanced(
            retriever_nodes=[(mock_retriever, "CUSTOM_RETRIEVER_1")],
            synthesize_node=(mock_synthesizer, "CUSTOM_SYNTHESIZER"),
        )
        
        assert isinstance(compiled, CompiledGraph)
        
    def test_compile_advanced_empty_retrievers_error(self):
        """Test error handling for empty retriever nodes list."""
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
        
        with pytest.raises(ValueError, match="retriever_nodes must be non-empty"):
            self.rag_agent.compile_advanced(
                retriever_nodes=[],
                synthesize_node=mock_synthesizer,
            )
            
    def test_compile_advanced_invalid_retriever(self):
        """Test error handling for invalid retriever in advanced mode."""
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
        
        with pytest.raises(ValueError, match="retriever must be callable or ToolNode"):
            # Type ignore for intentional error testing
            self.rag_agent.compile_advanced(
                retriever_nodes=["not_callable"],  # type: ignore
                synthesize_node=mock_synthesizer,
            )


class TestRAGAgentHelperMethods:
    """Test the helper methods of RAGAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state = AgentState()
        self.rag_agent = RAGAgent[AgentState](state=self.state)
        
    def test_add_optional_node_with_function(self):
        """Test _add_optional_node with a function."""
        def mock_node(state: AgentState) -> AgentState:
            return state
            
        result = self.rag_agent._add_optional_node(
            mock_node,
            default_name="TEST_NODE",
            label="test",
        )
        
        assert result == "TEST_NODE"
        
    def test_add_optional_node_with_tuple(self):
        """Test _add_optional_node with a tuple."""
        def mock_node(state: AgentState) -> AgentState:
            return state
            
        result = self.rag_agent._add_optional_node(
            (mock_node, "CUSTOM_NAME"),
            default_name="TEST_NODE",
            label="test",
        )
        
        assert result == "CUSTOM_NAME"
        
    def test_add_optional_node_none(self):
        """Test _add_optional_node with None."""
        result = self.rag_agent._add_optional_node(
            None,
            default_name="TEST_NODE",
            label="test",
        )
        
        assert result is None
        
    def test_add_optional_node_invalid_function(self):
        """Test _add_optional_node with invalid function."""
        with pytest.raises(ValueError, match="test node must be callable"):
            # Type ignore for intentional error testing
            self.rag_agent._add_optional_node(
                "not_callable",  # type: ignore
                default_name="TEST_NODE",
                label="test",
            )
            
    def test_add_retriever_nodes_single(self):
        """Test _add_retriever_nodes with single retriever."""
        def mock_retriever(state: AgentState) -> AgentState:
            return state
            
        result = self.rag_agent._add_retriever_nodes([mock_retriever])
        
        assert result == ["RETRIEVE_1"]
        
    def test_add_retriever_nodes_multiple(self):
        """Test _add_retriever_nodes with multiple retrievers."""
        def mock_retriever1(state: AgentState) -> AgentState:
            return state
            
        def mock_retriever2(state: AgentState) -> AgentState:
            return state
            
        result = self.rag_agent._add_retriever_nodes([mock_retriever1, mock_retriever2])
        
        assert result == ["RETRIEVE_1", "RETRIEVE_2"]
        
    def test_add_retriever_nodes_with_names(self):
        """Test _add_retriever_nodes with custom names."""
        def mock_retriever(state: AgentState) -> AgentState:
            return state
            
        result = self.rag_agent._add_retriever_nodes([(mock_retriever, "CUSTOM_RETRIEVER")])
        
        assert result == ["CUSTOM_RETRIEVER"]
        
    def test_add_retriever_nodes_empty_error(self):
        """Test _add_retriever_nodes with empty list."""
        with pytest.raises(ValueError, match="retriever_nodes must be non-empty"):
            self.rag_agent._add_retriever_nodes([])
            
    def test_add_synthesize_node_function(self):
        """Test _add_synthesize_node with function."""
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
            
        result = self.rag_agent._add_synthesize_node(mock_synthesizer)
        
        assert result == "SYNTHESIZE"
        
    def test_add_synthesize_node_tuple(self):
        """Test _add_synthesize_node with tuple."""
        def mock_synthesizer(state: AgentState) -> AgentState:
            return state
            
        result = self.rag_agent._add_synthesize_node((mock_synthesizer, "CUSTOM_SYNTH"))
        
        assert result == "CUSTOM_SYNTH"
        
    def test_add_synthesize_node_invalid(self):
        """Test _add_synthesize_node with invalid function."""
        with pytest.raises(ValueError, match="synthesize_node must be callable"):
            # Type ignore for intentional error testing
            self.rag_agent._add_synthesize_node("not_callable")  # type: ignore


class TestRAGAgentIntegration:
    """Integration tests for RAGAgent."""
    
    def test_basic_rag_execution_flow(self):
        """Test basic RAG execution flow."""
        def mock_retriever(state: AgentState) -> AgentState:
            # Simulate document retrieval
            retrieved_doc = Message.text_message("Retrieved: The capital of France is Paris.", role="system")
            state.context.append(retrieved_doc)
            return state
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            # Simulate answer synthesis based on retrieved content
            if state.context and len(state.context) > 0:
                # Check if the last message contains "Paris" using the text method
                last_message = state.context[-1]
                if hasattr(last_message, 'text') and "Paris" in last_message.text():
                    answer = Message.text_message("Based on the retrieved information, Paris is the capital of France.", role="assistant")
                    state.context.append(answer)
            return state
        
        rag_agent = RAGAgent[AgentState]()
        compiled = rag_agent.compile(
            retriever_node=mock_retriever,
            synthesize_node=mock_synthesizer,
        )
        
        # Execute the RAG agent
        initial_state = {"messages": [Message.text_message("What is the capital of France?", role="user")]}
        result = compiled.invoke(initial_state, config={"thread_id": "rag_test"})
        
        assert isinstance(result, dict)
        assert "messages" in result
        
    @pytest.mark.asyncio
    async def test_rag_async_execution(self):
        """Test RAG agent with async execution."""
        async def mock_async_retriever(state: AgentState) -> AgentState:
            retrieved_doc = Message.text_message("Async retrieved document", role="system")
            state.context.append(retrieved_doc)
            return state
            
        def mock_synthesizer(state: AgentState) -> AgentState:
            answer = Message.text_message("Async synthesized answer", role="assistant")
            state.context.append(answer)
            return state
        
        rag_agent = RAGAgent[AgentState]()
        compiled = rag_agent.compile(
            retriever_node=mock_async_retriever,
            synthesize_node=mock_synthesizer,
        )
        
        # Execute the RAG agent asynchronously
        initial_state = {"messages": [Message.text_message("Test async question", role="user")]}
        result = await compiled.ainvoke(initial_state, config={"thread_id": "async_rag_test"})
        
        assert isinstance(result, dict)
        assert "messages" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])