"""Comprehensive tests for QdrantStore implementation."""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict, List
from datetime import datetime
import json

# Try to import qdrant dependencies for testing
pytest_plugins = ("pytest_asyncio",)

try:
    from pyagenity.store.qdrant_store import (
        QdrantStore,
        QdrantStoreFactory,
        MemoryItem,
    )
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        CollectionInfo,
        CollectionsResponse,
        SearchResult,
        ScoredPoint,
    )
    qdrant_available = True
except ImportError:
    qdrant_available = False


# Skip all tests if qdrant is not available
pytestmark = pytest.mark.skipif(
    not qdrant_available,
    reason="qdrant-client not installed"
)


@pytest.fixture
def mock_embedding_function():
    """Mock embedding function that returns consistent embeddings."""
    def _embedding_fn(texts: List[str]) -> List[List[float]]:
        # Return mock embeddings - different for different texts for testing
        embeddings = []
        for i, text in enumerate(texts):
            # Create a deterministic embedding based on text hash
            base_embedding = [0.1 + (i * 0.1)] * 768
            embeddings.append(base_embedding)
        return embeddings
    return _embedding_fn


@pytest.fixture
async def mock_async_embedding_function():
    """Mock async embedding function."""
    async def _async_embedding_fn(texts: List[str]) -> List[List[float]]:
        # Simulate async embedding generation
        await asyncio.sleep(0.01)  # Small delay to simulate async work
        embeddings = []
        for i, text in enumerate(texts):
            base_embedding = [0.1 + (i * 0.1)] * 768
            embeddings.append(base_embedding)
        return embeddings
    return _async_embedding_fn


@pytest.fixture
def mock_async_client():
    """Mock AsyncQdrantClient."""
    client = AsyncMock(spec=AsyncQdrantClient)
    
    # Mock collections response
    collections_response = CollectionsResponse(collections=[])
    client.get_collections.return_value = collections_response
    
    # Mock successful operations
    client.create_collection.return_value = None
    client.upsert.return_value = None
    client.delete_collection.return_value = None
    client.close.return_value = None
    
    # Mock scroll response (empty by default)
    client.scroll.return_value = ([], None)
    
    # Mock search response (empty by default) 
    client.search.return_value = []
    
    return client


@pytest.fixture
def config():
    """Sample config for testing."""
    return {"conversation_id": "test_conv_123", "user_id": "test_user_456"}


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "content": "User prefers morning meetings",
        "type": "preference",
        "priority": "high"
    }


class TestMemoryItem:
    """Test MemoryItem dataclass."""

    def test_memory_item_creation(self):
        """Test creating a MemoryItem."""
        item = MemoryItem(
            id="test_id",
            content="test content",
            metadata={"key": "value"},
            timestamp="2024-01-01T00:00:00",
            embedding=[0.1, 0.2, 0.3]
        )
        
        assert item.id == "test_id"
        assert item.content == "test content"
        assert item.metadata == {"key": "value"}
        assert item.timestamp == "2024-01-01T00:00:00"
        assert item.embedding == [0.1, 0.2, 0.3]

    def test_memory_item_to_dict(self):
        """Test converting MemoryItem to dictionary."""
        item = MemoryItem(
            id="test_id",
            content="test content", 
            metadata={"key": "value"},
            timestamp="2024-01-01T00:00:00"
        )
        
        result = item.to_dict()
        expected = {
            "id": "test_id",
            "content": "test content",
            "metadata": {"key": "value"},
            "timestamp": "2024-01-01T00:00:00",
            "embedding": None
        }
        
        assert result == expected

    def test_memory_item_from_dict(self):
        """Test creating MemoryItem from dictionary."""
        data = {
            "id": "test_id", 
            "content": "test content",
            "metadata": {"key": "value"},
            "timestamp": "2024-01-01T00:00:00",
            "embedding": [0.1, 0.2, 0.3]
        }
        
        item = MemoryItem.from_dict(data)
        
        assert item.id == "test_id"
        assert item.content == "test content"
        assert item.metadata == {"key": "value"}
        assert item.timestamp == "2024-01-01T00:00:00"
        assert item.embedding == [0.1, 0.2, 0.3]


class TestQdrantStore:
    """Test QdrantStore implementation."""

    def test_init(self, mock_async_client, mock_embedding_function):
        """Test QdrantStore initialization."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function,
            collection_prefix="test_memory",
            vector_size=768,
            distance_metric=Distance.COSINE,
            max_retries=3,
            default_limit=10
        )
        
        assert store.async_client == mock_async_client
        assert store.embedding_function == mock_embedding_function
        assert store.collection_prefix == "test_memory"
        assert store.vector_size == 768
        assert store.distance_metric == Distance.COSINE
        assert store.max_retries == 3
        assert store.default_limit == 10
        assert store._collections_cache == set()

    def test_get_collection_name(self, mock_async_client, mock_embedding_function, config):
        """Test collection name generation."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        collection_name = store._get_collection_name(config)
        expected = "memory_test_conv_123_test_user_456"
        
        assert collection_name == expected

    def test_get_collection_name_long_identifier(self, mock_async_client, mock_embedding_function):
        """Test collection name generation with long identifiers."""
        store = QdrantStore(
            async_client=mock_async_client, 
            embedding_function=mock_embedding_function
        )
        
        long_config = {
            "conversation_id": "a" * 30,
            "user_id": "b" * 30,
            "session_id": "c" * 30
        }
        
        collection_name = store._get_collection_name(long_config)
        
        # Should be hashed due to length
        assert len(collection_name) < 50
        assert collection_name.startswith("memory_")

    def test_get_collection_name_default(self, mock_async_client, mock_embedding_function):
        """Test collection name generation with empty config."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        collection_name = store._get_collection_name({})
        
        assert collection_name == "memory_default"

    def test_extract_text_content_string(self, mock_async_client, mock_embedding_function):
        """Test extracting text from string data."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        content = store._extract_text_content("Hello world")
        assert content == "Hello world"

    def test_extract_text_content_dict_with_content(self, mock_async_client, mock_embedding_function):
        """Test extracting text from dict with content field.""" 
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        data = {"content": "Hello world", "other": "data"}
        content = store._extract_text_content(data)
        assert content == "Hello world"

    def test_extract_text_content_dict_without_content(self, mock_async_client, mock_embedding_function):
        """Test extracting text from dict without content field."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        data = {"message": "Hello world", "type": "info"}
        content = store._extract_text_content(data) 
        assert "Hello world" in content

    def test_extract_text_content_object(self, mock_async_client, mock_embedding_function):
        """Test extracting text from object with __dict__."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        class TestObj:
            def __init__(self):
                self.content = "Hello world"
                self.other = "data"
        
        obj = TestObj()
        content = store._extract_text_content(obj)
        assert "Hello world" in content

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_new(self, mock_async_client, mock_embedding_function):
        """Test ensuring collection exists when it doesn't."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        collection_name = "test_collection"
        
        # Mock empty collections response
        collections_response = CollectionsResponse(collections=[])
        mock_async_client.get_collections.return_value = collections_response
        
        await store._aensure_collection_exists(collection_name)
        
        # Should call create_collection
        mock_async_client.create_collection.assert_called_once()
        
        # Should be added to cache
        assert collection_name in store._collections_cache

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_cached(self, mock_async_client, mock_embedding_function):
        """Test ensuring collection exists when cached."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        collection_name = "test_collection"
        store._collections_cache.add(collection_name)
        
        await store._aensure_collection_exists(collection_name)
        
        # Should not call get_collections or create_collection
        mock_async_client.get_collections.assert_not_called()
        mock_async_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_acreate_memory_item(self, mock_async_client, mock_embedding_function, config, sample_data):
        """Test creating memory item from data."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        memory_item = await store._acreate_memory_item(config, sample_data)
        
        assert memory_item.content == "User prefers morning meetings"
        assert memory_item.metadata["type"] == "preference"
        assert memory_item.metadata["priority"] == "high"
        assert memory_item.metadata["config"] == config
        assert memory_item.embedding is not None
        assert len(memory_item.embedding) == 768
        assert isinstance(memory_item.id, str)
        assert isinstance(memory_item.timestamp, str)

    @pytest.mark.asyncio
    async def test_acreate_memory_item_async_embedding(self, mock_async_client, mock_async_embedding_function, config, sample_data):
        """Test creating memory item with async embedding function."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_async_embedding_function
        )
        
        memory_item = await store._acreate_memory_item(config, sample_data)
        
        assert memory_item.content == "User prefers morning meetings"
        assert memory_item.embedding is not None
        assert len(memory_item.embedding) == 768

    @pytest.mark.asyncio
    async def test_aupdate_memory(self, mock_async_client, mock_embedding_function, config, sample_data):
        """Test storing memory item."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        # Mock empty collections response
        collections_response = CollectionsResponse(collections=[])
        mock_async_client.get_collections.return_value = collections_response
        
        await store.aupdate_memory(config, sample_data)
        
        # Should ensure collection exists
        mock_async_client.get_collections.assert_called_once()
        mock_async_client.create_collection.assert_called_once()
        
        # Should upsert the point
        mock_async_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_memory_empty(self, mock_async_client, mock_embedding_function, config):
        """Test getting memory when none exists."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        # Mock empty scroll response
        mock_async_client.scroll.return_value = ([], None)
        
        result = await store.aget_memory(config)
        
        assert result is None
        mock_async_client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_aget_memory_with_data(self, mock_async_client, mock_embedding_function, config):
        """Test getting memory when data exists."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        # Mock point with payload
        mock_point = Mock()
        mock_point.payload = {"content": "test", "timestamp": "2024-01-01T00:00:00"}
        
        mock_async_client.scroll.return_value = ([mock_point], None)
        
        result = await store.aget_memory(config)
        
        assert result == {"content": "test", "timestamp": "2024-01-01T00:00:00"}
        mock_async_client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_adelete_memory(self, mock_async_client, mock_embedding_function, config):
        """Test deleting memory."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        collection_name = store._get_collection_name(config)
        store._collections_cache.add(collection_name)
        
        await store.adelete_memory(config)
        
        mock_async_client.delete_collection.assert_called_once_with(collection_name)
        assert collection_name not in store._collections_cache

    @pytest.mark.asyncio
    async def test_arelated_memory(self, mock_async_client, mock_embedding_function, config):
        """Test searching related memory."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        # Mock search results
        mock_result = Mock()
        mock_result.payload = {"content": "related content"}
        mock_async_client.search.return_value = [mock_result]
        
        results = await store.arelated_memory(config, "test query")
        
        assert len(results) == 1
        assert results[0] == {"content": "related content"}
        mock_async_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_arelease(self, mock_async_client, mock_embedding_function):
        """Test releasing resources."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        await store.arelease()
        
        mock_async_client.close.assert_called_once()

    def test_sync_update_memory(self, mock_async_client, mock_embedding_function, config, sample_data):
        """Test sync wrapper for update_memory."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        with patch.object(store, '_run_async') as mock_run_async:
            store.update_memory(config, sample_data)
            mock_run_async.assert_called_once()

    def test_sync_get_memory(self, mock_async_client, mock_embedding_function, config):
        """Test sync wrapper for get_memory."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        with patch.object(store, '_run_async') as mock_run_async:
            mock_run_async.return_value = {"test": "data"}
            
            result = store.get_memory(config)
            
            assert result == {"test": "data"}
            mock_run_async.assert_called_once()

    def test_sync_delete_memory(self, mock_async_client, mock_embedding_function, config):
        """Test sync wrapper for delete_memory."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        with patch.object(store, '_run_async') as mock_run_async:
            store.delete_memory(config)
            mock_run_async.assert_called_once()

    def test_sync_related_memory(self, mock_async_client, mock_embedding_function, config):
        """Test sync wrapper for related_memory."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        with patch.object(store, '_run_async') as mock_run_async:
            mock_run_async.return_value = [{"test": "data"}]
            
            result = store.related_memory(config, "test query")
            
            assert result == [{"test": "data"}]
            mock_run_async.assert_called_once()

    def test_sync_release(self, mock_async_client, mock_embedding_function):
        """Test sync wrapper for release."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        with patch.object(store, '_run_async') as mock_run_async:
            store.release()
            mock_run_async.assert_called_once()


class TestQdrantStoreFactory:
    """Test QdrantStoreFactory."""

    @pytest.mark.asyncio
    async def test_acreate_local_store(self, mock_embedding_function):
        """Test creating local store async."""
        with patch('pyagenity.store.qdrant_store.AsyncQdrantClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            store = await QdrantStoreFactory.acreate_local_store(
                embedding_function=mock_embedding_function,
                path="./test_data"
            )
            
            assert isinstance(store, QdrantStore)
            mock_client_class.assert_called_once_with(path="./test_data")

    @pytest.mark.asyncio
    async def test_acreate_remote_store(self, mock_embedding_function):
        """Test creating remote store async."""
        with patch('pyagenity.store.qdrant_store.AsyncQdrantClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            store = await QdrantStoreFactory.acreate_remote_store(
                embedding_function=mock_embedding_function,
                host="test.host",
                port=6333,
                api_key="test_key"
            )
            
            assert isinstance(store, QdrantStore)
            mock_client_class.assert_called_once_with(
                host="test.host", 
                port=6333, 
                api_key="test_key"
            )

    @pytest.mark.asyncio
    async def test_acreate_cloud_store(self, mock_embedding_function):
        """Test creating cloud store async.""" 
        with patch('pyagenity.store.qdrant_store.AsyncQdrantClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            store = await QdrantStoreFactory.acreate_cloud_store(
                embedding_function=mock_embedding_function,
                url="https://test.qdrant.cloud",
                api_key="test_key"
            )
            
            assert isinstance(store, QdrantStore)
            mock_client_class.assert_called_once_with(
                url="https://test.qdrant.cloud",
                api_key="test_key"
            )

    def test_create_local_store_sync(self, mock_embedding_function):
        """Test creating local store sync."""
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = Mock(spec=QdrantStore)
            
            store = QdrantStoreFactory.create_local_store(
                embedding_function=mock_embedding_function
            )
            
            assert mock_run.called
            # Verify the coroutine was passed to asyncio.run
            args, kwargs = mock_run.call_args
            assert len(args) == 1  # Should have one positional arg (the coroutine)

    def test_create_remote_store_sync(self, mock_embedding_function):
        """Test creating remote store sync."""
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = Mock(spec=QdrantStore)
            
            store = QdrantStoreFactory.create_remote_store(
                embedding_function=mock_embedding_function,
                host="test.host"
            )
            
            assert mock_run.called

    def test_create_cloud_store_sync(self, mock_embedding_function):
        """Test creating cloud store sync."""
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = Mock(spec=QdrantStore)
            
            store = QdrantStoreFactory.create_cloud_store(
                embedding_function=mock_embedding_function,
                url="https://test.qdrant.cloud",
                api_key="test_key"
            )
            
            assert mock_run.called


class TestQdrantStoreErrorHandling:
    """Test error handling in QdrantStore."""

    @pytest.mark.asyncio
    async def test_aupdate_memory_no_embedding(self, mock_async_client, config, sample_data):
        """Test update memory fails when embedding generation fails."""
        def failing_embedding_fn(texts):
            return [None]  # Return None to simulate failure
        
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=failing_embedding_fn
        )
        
        # Mock empty collections response
        collections_response = CollectionsResponse(collections=[])
        mock_async_client.get_collections.return_value = collections_response
        
        with pytest.raises(ValueError, match="Failed to generate embedding"):
            await store.aupdate_memory(config, sample_data)

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_error(self, mock_async_client, mock_embedding_function):
        """Test error handling in ensure_collection_exists."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        # Mock client to raise exception
        mock_async_client.get_collections.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            await store._aensure_collection_exists("test_collection")

    @pytest.mark.asyncio
    async def test_retry_operation_success_after_failure(self, mock_async_client, mock_embedding_function):
        """Test retry mechanism works."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function,
            max_retries=3
        )
        
        # Mock operation that fails twice then succeeds
        mock_operation = AsyncMock()
        mock_operation.side_effect = [
            Exception("Temporary failure"),
            Exception("Another failure"), 
            "Success"
        ]
        
        result = await store._aretry_operation(mock_operation)
        
        assert result == "Success"
        assert mock_operation.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_operation_max_retries_exceeded(self, mock_async_client, mock_embedding_function):
        """Test retry mechanism fails after max retries."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function,
            max_retries=2
        )
        
        # Mock operation that always fails
        mock_operation = AsyncMock()
        mock_operation.side_effect = Exception("Persistent failure")
        
        with pytest.raises(Exception, match="Persistent failure"):
            await store._aretry_operation(mock_operation)
        
        assert mock_operation.call_count == 2

    @pytest.mark.asyncio
    async def test_aget_memory_error(self, mock_async_client, mock_embedding_function, config):
        """Test error handling in aget_memory."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        # Mock client to raise exception
        mock_async_client.scroll.side_effect = Exception("Query failed")
        
        result = await store.aget_memory(config)
        
        assert result is None  # Should return None on error

    @pytest.mark.asyncio
    async def test_arelated_memory_error(self, mock_async_client, mock_embedding_function, config):
        """Test error handling in arelated_memory."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        # Mock client to raise exception
        mock_async_client.search.side_effect = Exception("Search failed")
        
        result = await store.arelated_memory(config, "test query")
        
        assert result == []  # Should return empty list on error

    @pytest.mark.asyncio
    async def test_arelease_error(self, mock_async_client, mock_embedding_function):
        """Test error handling in arelease."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        # Mock client to raise exception on close
        mock_async_client.close.side_effect = Exception("Close failed")
        
        # Should not raise exception, just log error
        await store.arelease()


class TestIntegrationScenarios:
    """Integration-style tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_complete_memory_lifecycle(self, mock_async_client, mock_embedding_function):
        """Test complete memory storage and retrieval lifecycle."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        config = {"conversation_id": "test_conv"}
        
        # Mock empty collections initially
        collections_response = CollectionsResponse(collections=[])
        mock_async_client.get_collections.return_value = collections_response
        
        # Test storing memory
        data1 = {"content": "User likes coffee", "type": "preference"}
        await store.aupdate_memory(config, data1)
        
        # Verify collection was created and point was upserted
        mock_async_client.create_collection.assert_called_once()
        mock_async_client.upsert.assert_called_once()
        
        # Mock point for retrieval
        mock_point = Mock()
        mock_point.payload = {"content": "User likes coffee", "timestamp": "2024-01-01T00:00:00"}
        mock_async_client.scroll.return_value = ([mock_point], None)
        
        # Test retrieving memory
        memory = await store.aget_memory(config)
        assert memory["content"] == "User likes coffee"
        
        # Mock search results
        mock_search_result = Mock()
        mock_search_result.payload = {"content": "User likes coffee"}
        mock_async_client.search.return_value = [mock_search_result]
        
        # Test searching related memory
        related = await store.arelated_memory(config, "beverage preferences")
        assert len(related) == 1
        assert related[0]["content"] == "User likes coffee"
        
        # Test deleting memory
        await store.adelete_memory(config)
        mock_async_client.delete_collection.assert_called_once()
        
        # Test cleanup
        await store.arelease()
        mock_async_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_configs_isolation(self, mock_async_client, mock_embedding_function):
        """Test that different configs create isolated collections."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        config1 = {"conversation_id": "conv1"}
        config2 = {"conversation_id": "conv2"}
        
        # Mock empty collections
        collections_response = CollectionsResponse(collections=[])
        mock_async_client.get_collections.return_value = collections_response
        
        # Store data for both configs
        await store.aupdate_memory(config1, {"content": "Data for conv1"})
        await store.aupdate_memory(config2, {"content": "Data for conv2"})
        
        # Should have created two collections
        assert mock_async_client.create_collection.call_count == 2
        
        # Verify different collection names
        call_args_list = mock_async_client.create_collection.call_args_list
        collection_names = [args[1]["collection_name"] for args in call_args_list]
        
        assert "memory_conv1" in collection_names
        assert "memory_conv2" in collection_names
        assert len(set(collection_names)) == 2  # Ensure they are different

    def test_sync_async_consistency(self, mock_async_client, mock_embedding_function, config, sample_data):
        """Test that sync and async methods produce consistent results."""
        store = QdrantStore(
            async_client=mock_async_client,
            embedding_function=mock_embedding_function
        )
        
        # Mock the async method to return a known value
        with patch.object(store, 'aget_memory', new_callable=AsyncMock) as mock_async_get:
            mock_async_get.return_value = {"test": "data"}
            
            # The sync method should wrap the async method
            with patch.object(store, '_run_async') as mock_run_async:
                mock_run_async.return_value = {"test": "data"}
                
                result = store.get_memory(config)
                
                assert result == {"test": "data"}
                mock_run_async.assert_called_once()