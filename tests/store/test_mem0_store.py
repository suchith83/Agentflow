"""
Comprehensive tests for the Mem0Store implementation.

This test file validates the Mem0Store class and its integration with Mem0
using pytest, including both sync and async test patterns, mocking for
external dependencies, and edge cases.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from uuid import uuid4

from pyagenity.store.mem0_store import Mem0Store, create_mem0_store, create_mem0_store_with_qdrant
from pyagenity.store.base_store import MemorySearchResult, MemoryRecord
from pyagenity.utils.message import Message, TextBlock, TokenUsages


# Test fixtures and mock classes

class MockMem0:
    """Mock Mem0 Memory class for testing."""
    
    def __init__(self):
        self.memories = []
        self.next_id = 1
    
    def add(self, messages, user_id, metadata=None, **kwargs):
        """Mock add method."""
        memory_id = f"mem0_{self.next_id}"
        self.next_id += 1
        
        # Simulate Mem0's response format
        memory_data = {
            "id": memory_id,
            "memory": messages[0]["content"] if messages else "",
            "metadata": metadata or {},
            "user_id": user_id,
        }
        
        # Store metadata with mem0_id for testing
        if metadata:
            metadata["mem0_id"] = memory_id
            memory_data["metadata"] = metadata
        
        self.memories.append(memory_data)
        
        return {
            "results": [{"id": memory_id}],
            "id": memory_id
        }
    
    def search(self, query, user_id, limit=10, filters=None, **kwargs):
        """Mock search method."""
        results = []
        
        for memory in self.memories:
            if memory["user_id"] != user_id:
                continue
                
            # Apply filters
            if filters:
                metadata = memory.get("metadata", {})
                match = True
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Improved text search - exact match or word-level matching
            if not query or query.lower() in memory["memory"].lower():
                results.append({
                    "id": memory["id"],
                    "memory": memory["memory"],
                    "score": 0.85,  # Mock score
                    "metadata": memory.get("metadata", {})
                })
            else:
                # For more realistic search, also try word-level matching
                query_words = set(query.lower().split())
                memory_words = set(memory["memory"].lower().split())
                if query_words & memory_words:  # If any query words match memory words
                    results.append({
                        "id": memory["id"],
                        "memory": memory["memory"],
                        "score": 0.75,  # Slightly lower score for word matches
                        "metadata": memory.get("metadata", {})
                    })
        
        return {"results": results[:limit]}
    
    def get_all(self, user_id, **kwargs):
        """Mock get_all method."""
        return [mem for mem in self.memories if mem["user_id"] == user_id]
    
    def update(self, memory_id, data=None, metadata=None, **kwargs):
        """Mock update method."""
        for memory in self.memories:
            if memory["id"] == memory_id:
                if data:
                    memory["memory"] = data
                if metadata:
                    memory["metadata"] = metadata
                return True
        return False
    
    def delete(self, memory_id, **kwargs):
        """Mock delete method."""
        self.memories = [mem for mem in self.memories if mem["id"] != memory_id]
        return True
    
    @classmethod
    def from_config(cls, config):
        """Mock from_config class method."""
        return cls()


@pytest.fixture
def mock_mem0():
    """Provide a MockMem0 instance."""
    return MockMem0()


@pytest.fixture
def mem0_store(mock_mem0):
    """Provide a Mem0Store instance with mocked Mem0."""
    with patch('pyagenity.store.mem0_store.Memory', return_value=mock_mem0):
        store = Mem0Store(
            user_id="test_user",
            agent_id="test_agent",
            app_id="test_app"
        )
        return store


@pytest.fixture
def sample_message():
    """Provide a sample Message object."""
    return Message(
        role="user",
        content=[TextBlock(text="Hello, this is a test message")],
        message_id=str(uuid4()),
        timestamp=datetime.now(),
        usages=TokenUsages(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )


# Core functionality tests

class TestMem0StoreInitialization:
    """Test Mem0Store initialization and configuration."""
    
    def test_init_with_defaults(self, mock_mem0):
        """Test initialization with default parameters."""
        with patch('pyagenity.store.mem0_store.Memory', return_value=mock_mem0):
            store = Mem0Store()
            
            assert store.default_user_id == "default_user"
            assert store.default_agent_id is None
            assert store.app_id == "pyagenity_app"
            assert store.config == {}
    
    def test_init_with_custom_params(self, mock_mem0):
        """Test initialization with custom parameters."""
        config = {"llm": {"provider": "openai"}}
        
        with patch('pyagenity.store.mem0_store.Memory', return_value=mock_mem0):
            store = Mem0Store(
                config=config,
                user_id="custom_user",
                agent_id="custom_agent",
                app_id="custom_app"
            )
            
            assert store.default_user_id == "custom_user"
            assert store.default_agent_id == "custom_agent"
            assert store.app_id == "custom_app"
            assert store.config == config
    
    def test_init_with_config(self, mock_mem0):
        """Test initialization with Mem0 config."""
        config = {
            "vector_store": {"provider": "qdrant"},
            "llm": {"provider": "openai"}
        }
        
        with patch('pyagenity.store.mem0_store.Memory') as mock_memory:
            mock_memory.from_config.return_value = mock_mem0
            
            store = Mem0Store(config=config)
            
            mock_memory.from_config.assert_called_once_with(config)
            assert store.config == config
    
    def test_init_failure(self):
        """Test initialization failure handling."""
        with patch('pyagenity.store.mem0_store.Memory', side_effect=Exception("Init failed")):
            with pytest.raises(RuntimeError, match="Failed to initialize Mem0 Memory"):
                Mem0Store()


class TestMem0StoreCoreOperations:
    """Test core CRUD operations."""
    
    def test_add_memory_success(self, mem0_store):
        """Test successful memory addition."""
        content = "This is a test memory"
        
        memory_id = mem0_store.add(content)
        
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0
        
        # Verify the memory was actually added
        stored_memory = mem0_store.get(memory_id)
        assert stored_memory is not None
    
    def test_add_memory_with_metadata(self, mem0_store):
        """Test memory addition with metadata."""
        content = "Test memory with metadata"
        metadata = {"custom_field": "custom_value", "importance": "high"}
        
        memory_id = mem0_store.add(
            content,
            user_id="specific_user",
            agent_id="specific_agent",
            memory_type="semantic",
            category="facts",
            metadata=metadata
        )
        
        stored_memory = mem0_store.get(memory_id)
        assert stored_memory is not None
        assert stored_memory.content == content
        assert stored_memory.memory_type == "semantic"
        assert stored_memory.user_id == "specific_user"
        assert stored_memory.agent_id == "specific_agent"
        assert "custom_field" in stored_memory.metadata
        assert stored_memory.metadata["custom_field"] == "custom_value"
    
    def test_add_memory_empty_content(self, mem0_store):
        """Test adding memory with empty content raises error."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            mem0_store.add("")
        
        with pytest.raises(ValueError, match="Content cannot be empty"):
            mem0_store.add("   ")
    
    def test_search_memories(self, mem0_store):
        """Test memory search functionality."""
        # Add some test memories
        mem0_store.add("Python programming is great")
        mem0_store.add("Machine learning with PyTorch")
        mem0_store.add("Data science project")
        
        # Search for memories
        results = mem0_store.search("Python")
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert any("Python" in result.content for result in results)
    
    def test_search_with_filters(self, mem0_store):
        """Test search with various filters."""
        # Add memories with different types and agents
        mem0_store.add(
            "Episodic memory", 
            agent_id="agent1", 
            memory_type="episodic",
            category="conversation"
        )
        mem0_store.add(
            "Semantic memory", 
            agent_id="agent2", 
            memory_type="semantic",
            category="facts"
        )
        
        # Search with agent filter
        results = mem0_store.search("memory", agent_id="agent1")
        assert len(results) == 1
        assert results[0].agent_id == "agent1"
        
        # Search with memory type filter
        results = mem0_store.search("memory", memory_type="semantic")
        assert len(results) == 1
        assert results[0].memory_type == "semantic"
    
    def test_get_memory_by_id(self, mem0_store):
        """Test retrieving memory by ID."""
        content = "Specific memory to retrieve"
        memory_id = mem0_store.add(content)
        
        retrieved = mem0_store.get(memory_id)
        
        assert retrieved is not None
        assert retrieved.id == memory_id
        assert retrieved.content == content
    
    def test_get_nonexistent_memory(self, mem0_store):
        """Test retrieving non-existent memory returns None."""
        result = mem0_store.get("nonexistent_id")
        assert result is None
    
    def test_update_memory(self, mem0_store):
        """Test memory update functionality."""
        # Add initial memory
        memory_id = mem0_store.add("Original content")
        
        # Update content
        new_content = "Updated content"
        mem0_store.update(memory_id, content=new_content)
        
        # Verify update
        updated = mem0_store.get(memory_id)
        assert updated is not None
        assert updated.content == new_content
    
    def test_update_memory_metadata(self, mem0_store):
        """Test updating memory metadata."""
        # Add memory with initial metadata
        initial_metadata = {"version": 1, "type": "test"}
        memory_id = mem0_store.add("Test content", metadata=initial_metadata)
        
        # Update metadata
        new_metadata = {"version": 2, "updated": True}
        mem0_store.update(memory_id, metadata=new_metadata)
        
        # Verify metadata update
        updated = mem0_store.get(memory_id)
        assert updated is not None
        assert updated.metadata["version"] == 2
        assert updated.metadata["updated"] is True
    
    def test_update_nonexistent_memory(self, mem0_store):
        """Test updating non-existent memory raises error."""
        with pytest.raises(RuntimeError, match="Update failed: Memory with ID nonexistent not found"):
            mem0_store.update("nonexistent", content="New content")
    
    def test_delete_memory(self, mem0_store):
        """Test memory deletion."""
        # Add memory
        memory_id = mem0_store.add("Memory to delete")
        
        # Verify it exists
        assert mem0_store.get(memory_id) is not None
        
        # Delete memory
        mem0_store.delete(memory_id)
        
        # Verify it's gone
        assert mem0_store.get(memory_id) is None
    
    def test_delete_nonexistent_memory(self, mem0_store):
        """Test deleting non-existent memory (should not raise error)."""
        # Should not raise an error
        mem0_store.delete("nonexistent_id")


class TestMem0StoreAsyncOperations:
    """Test async versions of operations."""
    
    @pytest.mark.asyncio
    async def test_async_add(self, mem0_store):
        """Test async memory addition."""
        content = "Async test memory"
        
        memory_id = await mem0_store.aadd(content)
        
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0
    
    @pytest.mark.asyncio
    async def test_async_search(self, mem0_store):
        """Test async memory search."""
        # Add test memory
        await mem0_store.aadd("Async searchable content")
        
        # Search asynchronously
        results = await mem0_store.asearch("searchable")
        
        assert isinstance(results, list)
        assert len(results) > 0
    
    @pytest.mark.asyncio
    async def test_async_get(self, mem0_store):
        """Test async memory retrieval."""
        memory_id = await mem0_store.aadd("Async retrievable memory")
        
        retrieved = await mem0_store.aget(memory_id)
        
        assert retrieved is not None
        assert retrieved.id == memory_id
    
    @pytest.mark.asyncio
    async def test_async_update(self, mem0_store):
        """Test async memory update."""
        memory_id = await mem0_store.aadd("Original async content")
        
        await mem0_store.aupdate(memory_id, content="Updated async content")
        
        updated = await mem0_store.aget(memory_id)
        assert updated is not None
        assert updated.content == "Updated async content"
    
    @pytest.mark.asyncio
    async def test_async_delete(self, mem0_store):
        """Test async memory deletion."""
        memory_id = await mem0_store.aadd("Memory to delete async")
        
        await mem0_store.adelete(memory_id)
        
        deleted = await mem0_store.aget(memory_id)
        assert deleted is None


class TestMem0StoreBulkOperations:
    """Test bulk operations like get_all and delete_all."""
    
    def test_get_all_memories(self, mem0_store):
        """Test retrieving all memories."""
        # Add multiple memories
        mem0_store.add("Memory 1", memory_type="episodic")
        mem0_store.add("Memory 2", memory_type="semantic")
        mem0_store.add("Memory 3", agent_id="specific_agent")
        
        # Get all memories
        all_memories = mem0_store.get_all()
        
        assert isinstance(all_memories, list)
        assert len(all_memories) >= 3
    
    def test_get_all_with_filters(self, mem0_store):
        """Test get_all with filters."""
        # Add memories with different properties
        mem0_store.add("Agent 1 memory", agent_id="agent1")
        mem0_store.add("Agent 2 memory", agent_id="agent2")
        mem0_store.add("Semantic memory", memory_type="semantic")
        
        # Filter by agent
        agent1_memories = mem0_store.get_all(agent_id="agent1")
        assert len(agent1_memories) == 1
        assert agent1_memories[0].agent_id == "agent1"
        
        # Filter by memory type
        semantic_memories = mem0_store.get_all(memory_type="semantic")
        assert len(semantic_memories) == 1
        assert semantic_memories[0].memory_type == "semantic"
    
    def test_delete_all_memories(self, mem0_store):
        """Test deleting all memories."""
        # Add multiple memories
        mem0_store.add("Memory 1")
        mem0_store.add("Memory 2")
        mem0_store.add("Memory 3")
        
        # Delete all memories
        deleted_count = mem0_store.delete_all()
        
        assert deleted_count >= 3
        
        # Verify memories are gone
        remaining = mem0_store.get_all()
        assert len(remaining) == 0
    
    def test_delete_all_with_filters(self, mem0_store):
        """Test delete_all with filters."""
        # Add memories with different agents
        mem0_store.add("Keep this", agent_id="keep_agent")
        mem0_store.add("Delete this 1", agent_id="delete_agent")
        mem0_store.add("Delete this 2", agent_id="delete_agent")
        
        # Delete only memories for specific agent
        deleted_count = mem0_store.delete_all(agent_id="delete_agent")
        assert deleted_count == 2
        
        # Verify correct memories remain
        remaining = mem0_store.get_all()
        assert len(remaining) == 1
        assert remaining[0].agent_id == "keep_agent"
    
    @pytest.mark.asyncio
    async def test_async_get_all(self, mem0_store):
        """Test async get_all operation."""
        await mem0_store.aadd("Async memory 1")
        await mem0_store.aadd("Async memory 2")
        
        all_memories = await mem0_store.aget_all()
        
        assert isinstance(all_memories, list)
        assert len(all_memories) >= 2
    
    @pytest.mark.asyncio
    async def test_async_delete_all(self, mem0_store):
        """Test async delete_all operation."""
        await mem0_store.aadd("Memory to delete 1")
        await mem0_store.aadd("Memory to delete 2")
        
        deleted_count = await mem0_store.adelete_all()
        
        assert deleted_count >= 2


class TestMem0StoreMessageIntegration:
    """Test integration with PyAgenity Message objects."""
    
    def test_store_message(self, mem0_store, sample_message):
        """Test storing a Message object."""
        memory_id = mem0_store.store_message(
            sample_message,
            additional_metadata={"conversation_id": "conv_123"}
        )
        
        assert isinstance(memory_id, str)
        assert len(memory_id) > 0
        
        # Retrieve and verify
        stored = mem0_store.get(memory_id)
        assert stored is not None
        assert "Hello, this is a test message" in stored.content
        assert stored.metadata["role"] == "user"
        assert stored.metadata["conversation_id"] == "conv_123"
    
    @pytest.mark.asyncio
    async def test_async_store_message(self, mem0_store, sample_message):
        """Test async message storage."""
        memory_id = await mem0_store.astore_message(
            sample_message,
            user_id="async_user",
            agent_id="async_agent"
        )
        
        assert isinstance(memory_id, str)
        
        stored = await mem0_store.aget(memory_id)
        assert stored is not None
        assert stored.user_id == "async_user"
        assert stored.agent_id == "async_agent"
    
    def test_recall_similar_messages(self, mem0_store, sample_message):
        """Test recalling similar messages."""
        # Store multiple messages
        message1 = Message(role="user", content=[TextBlock(text="I need help with Python")])
        message2 = Message(role="assistant", content=[TextBlock(text="Sure, I can help with Python programming")])
        message3 = Message(role="user", content=[TextBlock(text="Tell me about JavaScript")])
        
        mem0_store.store_message(message1)
        mem0_store.store_message(message2)
        mem0_store.store_message(message3)
        
        # Recall messages similar to Python query
        similar = mem0_store.recall_similar_messages("Python programming help", limit=5)
        
        assert isinstance(similar, list)
        assert len(similar) > 0
        assert any("Python" in result.content for result in similar)
    
    def test_recall_with_role_filter(self, mem0_store):
        """Test recalling messages with role filter."""
        # Store messages with different roles
        user_msg = Message(role="user", content=[TextBlock(text="User question about coding")])
        assistant_msg = Message(role="assistant", content=[TextBlock(text="Assistant response about coding")])
        
        mem0_store.store_message(user_msg)
        mem0_store.store_message(assistant_msg)
        
        # Recall only user messages
        user_memories = mem0_store.recall_similar_messages(
            "coding", 
            role_filter="user"
        )
        
        assert len(user_memories) == 1
        assert user_memories[0].metadata["role"] == "user"


class TestMem0StoreStatistics:
    """Test statistics and utility methods."""
    
    def test_get_stats_empty(self, mem0_store):
        """Test statistics for empty store."""
        stats = mem0_store.get_stats()
        
        assert isinstance(stats, dict)
        assert stats["total_memories"] == 0
        assert stats["user_id"] == "test_user"
        assert stats["app_id"] == "test_app"
    
    def test_get_stats_with_data(self, mem0_store):
        """Test statistics with actual data."""
        # Add memories with different types and categories
        mem0_store.add("Memory 1", memory_type="episodic", category="conversation")
        mem0_store.add("Memory 2", memory_type="semantic", category="facts")
        mem0_store.add("Memory 3", memory_type="episodic", category="conversation")
        
        stats = mem0_store.get_stats()
        
        assert stats["total_memories"] == 3
        assert "episodic" in stats["memory_types"]
        assert "semantic" in stats["memory_types"]
        assert stats["memory_types"]["episodic"] == 2
        assert stats["memory_types"]["semantic"] == 1
        assert "conversation" in stats["categories"]
        assert "facts" in stats["categories"]
    
    @pytest.mark.asyncio
    async def test_async_get_stats(self, mem0_store):
        """Test async statistics retrieval."""
        await mem0_store.aadd("Async memory for stats")
        
        stats = await mem0_store.aget_stats()
        
        assert isinstance(stats, dict)
        assert stats["total_memories"] >= 1
    
    def test_cleanup(self, mem0_store):
        """Test cleanup operation."""
        # Should not raise an error
        mem0_store.cleanup()
    
    @pytest.mark.asyncio
    async def test_async_cleanup(self, mem0_store):
        """Test async cleanup."""
        await mem0_store.acleanup()


class TestMem0StoreFactoryFunctions:
    """Test factory functions for creating Mem0Store instances."""
    
    def test_create_mem0_store_default(self):
        """Test creating Mem0Store with defaults."""
        with patch('pyagenity.store.mem0_store.Memory'):
            store = create_mem0_store()
            
            assert isinstance(store, Mem0Store)
            assert store.default_user_id == "default_user"
            assert store.app_id == "pyagenity_app"
    
    def test_create_mem0_store_with_params(self):
        """Test creating Mem0Store with custom parameters."""
        config = {"llm": {"provider": "openai"}}
        
        with patch('pyagenity.store.mem0_store.Memory'):
            store = create_mem0_store(
                config=config,
                user_id="custom_user",
                agent_id="custom_agent",
                app_id="custom_app"
            )
            
            assert store.config == config
            assert store.default_user_id == "custom_user"
            assert store.default_agent_id == "custom_agent"
            assert store.app_id == "custom_app"
    
    def test_create_mem0_store_with_qdrant(self):
        """Test creating Mem0Store with Qdrant configuration."""
        with patch('pyagenity.store.mem0_store.Memory') as mock_memory_class:
            # Mock the from_config class method
            mock_instance = Mock()
            mock_memory_class.from_config.return_value = mock_instance
            
            store = create_mem0_store_with_qdrant(
                qdrant_url="http://localhost:6333",
                qdrant_api_key="test_key",
                collection_name="test_collection",
                embedding_model="text-embedding-ada-002",
                llm_model="gpt-4",
                user_id="qdrant_user"
            )
            
            assert isinstance(store, Mem0Store)
            assert store.default_user_id == "qdrant_user"
            
            # Verify from_config was called with correct structure
            mock_memory_class.from_config.assert_called_once()
            call_args = mock_memory_class.from_config.call_args[0][0]
            assert call_args["vector_store"]["provider"] == "qdrant"
            assert call_args["vector_store"]["config"]["collection_name"] == "test_collection"
            assert call_args["embedder"]["provider"] == "openai"
            assert call_args["llm"]["provider"] == "openai"


class TestMem0StoreErrorHandling:
    """Test error handling and edge cases."""
    
    def test_mem0_search_failure(self, mem0_store):
        """Test handling of Mem0 search failures."""
        # Mock Mem0 to raise an exception during search
        mem0_store.memory.search = Mock(side_effect=Exception("Search failed"))
        
        with pytest.raises(RuntimeError, match="Search failed"):
            mem0_store.search("test query")
    
    def test_mem0_add_failure(self, mem0_store):
        """Test handling of Mem0 add failures."""
        # Mock Mem0 to raise an exception during add
        mem0_store.memory.add = Mock(side_effect=Exception("Add failed"))
        
        with pytest.raises(RuntimeError, match="Failed to add memory"):
            mem0_store.add("test content")
    
    def test_invalid_datetime_parsing(self, mem0_store):
        """Test handling of invalid datetime strings."""
        result = mem0_store._parse_datetime("invalid_datetime")
        assert result is None
        
        result = mem0_store._parse_datetime(None)
        assert result is None
    
    def test_get_stats_error(self, mem0_store):
        """Test statistics error handling."""
        # Mock get_all to raise an exception
        mem0_store.get_all = Mock(side_effect=Exception("Stats failed"))
        
        stats = mem0_store.get_stats()
        
        assert isinstance(stats, dict)
        assert "error" in stats
        assert "Stats failed" in stats["error"]
    
    def test_score_threshold_filtering(self, mem0_store):
        """Test score threshold filtering in search results."""
        # This would require mocking Mem0 to return specific scores
        mem0_store.add("Test memory")
        
        # Mock search to return results with specific scores
        mock_results = {
            "results": [
                {"id": "1", "memory": "High score memory", "score": 0.9, "metadata": {"app_id": "test_app"}},
                {"id": "2", "memory": "Low score memory", "score": 0.3, "metadata": {"app_id": "test_app"}}
            ]
        }
        mem0_store.memory.search = Mock(return_value=mock_results)
        
        # Search with score threshold
        results = mem0_store.search("memory", score_threshold=0.5)

        print("results: ", results)
        
        # Only high score result should be returned
        assert len(results) == 1
        assert results[0].content == "High score memory"
        assert results[0].score == 0.9


class TestMem0StoreUtilities:
    """Test utility methods and helper functions."""
    
    def test_parse_datetime_valid(self, mem0_store):
        """Test parsing valid datetime strings."""
        # Test ISO format
        iso_string = "2023-12-01T10:30:00"
        parsed = mem0_store._parse_datetime(iso_string)
        assert parsed is not None
        assert isinstance(parsed, datetime)
        
        # Test with timezone
        iso_string_tz = "2023-12-01T10:30:00+00:00"
        parsed_tz = mem0_store._parse_datetime(iso_string_tz)
        assert parsed_tz is not None
        
        # Test Z suffix (UTC)
        iso_string_z = "2023-12-01T10:30:00Z"
        parsed_z = mem0_store._parse_datetime(iso_string_z)
        assert parsed_z is not None
    
    def test_parse_datetime_invalid(self, mem0_store):
        """Test parsing invalid datetime strings."""
        # Invalid format
        invalid = mem0_store._parse_datetime("not a datetime")
        assert invalid is None
        
        # None input
        none_result = mem0_store._parse_datetime(None)
        assert none_result is None
        
        # Empty string
        empty_result = mem0_store._parse_datetime("")
        assert empty_result is None


# Integration tests (would require actual Mem0 setup)

@pytest.mark.integration
class TestMem0StoreIntegration:
    """Integration tests that would require actual Mem0 setup."""
    
    @pytest.mark.skip(reason="Requires actual Mem0 installation and configuration")
    def test_real_mem0_integration(self):
        """Test with real Mem0 instance."""
        # This would test with actual Mem0 installation
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "url": "http://localhost:6333",
                    "collection_name": "test_collection"
                }
            }
        }
        
        store = Mem0Store(config=config, user_id="integration_test")
        
        # Add memory
        memory_id = store.add("Integration test memory")
        assert memory_id is not None
        
        # Search memory
        results = store.search("Integration test")
        assert len(results) > 0
        
        # Clean up
        store.delete(memory_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])