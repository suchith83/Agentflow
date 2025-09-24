"""
Comprehensive tests for QdrantVectorStore implementation.

These tests cover:
- Collection management (create, list, delete, exists)
- Vector operations (insert, search, get, update, delete)
- Message-specific operations (store_message, recall_similar_messages)
- Utility methods (stats, cleanup)
- Error handling and edge cases
- Both sync and async operations
"""

import asyncio
import pytest
import uuid
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from pyagenity.store.qdrant_vector_store import QdrantVectorStore
from pyagenity.store.vector_base_store import (
    DistanceMetric,
    VectorSearchResult,
    MemoryRecord
)
from pyagenity.utils.message import Message


# Test fixtures and mocks

@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient for testing."""
    mock_client = Mock()
    
    # Collection methods
    mock_client.create_collection = Mock()
    mock_client.get_collections = Mock()
    mock_client.delete_collection = Mock()
    mock_client.get_collection = Mock()
    
    # Vector operations
    mock_client.upsert = Mock()
    mock_client.search = Mock()
    mock_client.retrieve = Mock()
    mock_client.delete = Mock()
    mock_client.set_payload = Mock()
    
    return mock_client


@pytest.fixture
def mock_async_qdrant_client():
    """Mock AsyncQdrantClient for testing."""
    mock_client = AsyncMock()
    
    # Collection methods
    mock_client.create_collection = AsyncMock()
    mock_client.get_collections = AsyncMock()
    mock_client.delete_collection = AsyncMock()
    mock_client.get_collection = AsyncMock()
    
    # Vector operations
    mock_client.upsert = AsyncMock()
    mock_client.search = AsyncMock()
    mock_client.retrieve = AsyncMock()
    mock_client.delete = AsyncMock()
    mock_client.set_payload = AsyncMock()
    
    return mock_client


@pytest.fixture
def mock_collections_response():
    """Mock response for get_collections."""
    mock_collection = Mock()
    mock_collection.name = "test_collection"
    
    mock_response = Mock()
    mock_response.collections = [mock_collection]
    
    return mock_response


@pytest.fixture
def mock_point():
    """Mock Qdrant point for testing."""
    point = Mock()
    point.id = "test_id"
    point.score = 0.95
    point.payload = {"content": "test content", "role": "user"}
    point.vector = [0.1, 0.2, 0.3]
    return point


@pytest.fixture
def sample_message():
    """Create a sample Message for testing."""
    return Message.text_message("Hello, this is a test message", role="user")


@pytest.fixture
def sample_embedding():
    """Sample embedding vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


class TestQdrantVectorStoreInit:
    """Test QdrantVectorStore initialization."""
    
    @patch('pyagenity.store.qdrant_vector_store.QdrantClient')
    @patch('pyagenity.store.qdrant_vector_store.AsyncQdrantClient')
    def test_init_local_path(self, mock_async_client_cls, mock_client_cls):
        """Test initialization with local path."""
        store = QdrantVectorStore(path="./test_data")
        
        mock_client_cls.assert_called_once_with(path="./test_data")
        mock_async_client_cls.assert_called_once_with(path="./test_data")
        assert store._collection_cache == set()
    
    @patch('pyagenity.store.qdrant_vector_store.QdrantClient')
    @patch('pyagenity.store.qdrant_vector_store.AsyncQdrantClient')
    def test_init_remote_host(self, mock_async_client_cls, mock_client_cls):
        """Test initialization with remote host."""
        store = QdrantVectorStore(host="remote-host", port=6333)
        
        mock_client_cls.assert_called_once_with(host="remote-host", port=6333, api_key=None)
        mock_async_client_cls.assert_called_once_with(host="remote-host", port=6333, api_key=None)
    
    @patch('pyagenity.store.qdrant_vector_store.QdrantClient')
    @patch('pyagenity.store.qdrant_vector_store.AsyncQdrantClient')
    def test_init_cloud_url(self, mock_async_client_cls, mock_client_cls):
        """Test initialization with cloud URL."""
        store = QdrantVectorStore(url="https://test.qdrant.io", api_key="test-key")
        
        mock_client_cls.assert_called_once_with(url="https://test.qdrant.io", api_key="test-key")
        mock_async_client_cls.assert_called_once_with(url="https://test.qdrant.io", api_key="test-key")
    
    @patch('pyagenity.store.qdrant_vector_store.QdrantClient')
    @patch('pyagenity.store.qdrant_vector_store.AsyncQdrantClient')
    def test_init_defaults(self, mock_async_client_cls, mock_client_cls):
        """Test initialization with defaults."""
        store = QdrantVectorStore()
        
        mock_client_cls.assert_called_once_with(host="localhost", port=6333, api_key=None)
        mock_async_client_cls.assert_called_once_with(host="localhost", port=6333, api_key=None)


class TestQdrantVectorStoreCollections:
    """Test collection management operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('pyagenity.store.qdrant_vector_store.QdrantClient'), \
             patch('pyagenity.store.qdrant_vector_store.AsyncQdrantClient'):
            self.store = QdrantVectorStore(path="./test_data")
    
    def test_create_collection_success(self, mock_qdrant_client):
        """Test successful collection creation."""
        self.store.client = mock_qdrant_client
        
        self.store.create_collection("test_collection", 512, DistanceMetric.COSINE)
        
        mock_qdrant_client.create_collection.assert_called_once()
        assert "test_collection" in self.store._collection_cache
    
    def test_create_collection_already_exists(self, mock_qdrant_client):
        """Test creating collection that already exists."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.create_collection.side_effect = Exception("Collection already exists")
        
        # Should not raise exception
        self.store.create_collection("test_collection", 512)
        assert "test_collection" in self.store._collection_cache
    
    def test_create_collection_error(self, mock_qdrant_client):
        """Test collection creation error."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.create_collection.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            self.store.create_collection("test_collection", 512)
    
    @pytest.mark.asyncio
    async def test_acreate_collection_success(self, mock_async_qdrant_client):
        """Test async collection creation."""
        self.store.async_client = mock_async_qdrant_client
        
        await self.store.acreate_collection("test_collection", 512, DistanceMetric.EUCLIDEAN)
        
        mock_async_qdrant_client.create_collection.assert_called_once()
        assert "test_collection" in self.store._collection_cache
    
    def test_list_collections(self, mock_qdrant_client, mock_collections_response):
        """Test listing collections."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.get_collections.return_value = mock_collections_response
        
        collections = self.store.list_collections()
        
        assert collections == ["test_collection"]
        assert "test_collection" in self.store._collection_cache
    
    @pytest.mark.asyncio
    async def test_alist_collections(self, mock_async_qdrant_client, mock_collections_response):
        """Test async listing collections."""
        self.store.async_client = mock_async_qdrant_client
        mock_async_qdrant_client.get_collections.return_value = mock_collections_response
        
        collections = await self.store.alist_collections()
        
        assert collections == ["test_collection"]
        assert "test_collection" in self.store._collection_cache
    
    def test_delete_collection(self, mock_qdrant_client):
        """Test collection deletion."""
        self.store.client = mock_qdrant_client
        self.store._collection_cache.add("test_collection")
        
        self.store.delete_collection("test_collection")
        
        mock_qdrant_client.delete_collection.assert_called_once_with(collection_name="test_collection")
        assert "test_collection" not in self.store._collection_cache
    
    @pytest.mark.asyncio
    async def test_adelete_collection(self, mock_async_qdrant_client):
        """Test async collection deletion."""
        self.store.async_client = mock_async_qdrant_client
        self.store._collection_cache.add("test_collection")
        
        await self.store.adelete_collection("test_collection")
        
        mock_async_qdrant_client.delete_collection.assert_called_once_with(collection_name="test_collection")
        assert "test_collection" not in self.store._collection_cache
    
    def test_collection_exists_cached(self):
        """Test collection exists check with cached result."""
        self.store._collection_cache.add("cached_collection")
        
        result = self.store.collection_exists("cached_collection")
        
        assert result is True
    
    def test_collection_exists_not_cached(self, mock_qdrant_client, mock_collections_response):
        """Test collection exists check without cache."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.get_collections.return_value = mock_collections_response
        
        result = self.store.collection_exists("test_collection")
        
        assert result is True
        assert "test_collection" in self.store._collection_cache
    
    @pytest.mark.asyncio
    async def test_acollection_exists_cached(self):
        """Test async collection exists with cached result."""
        self.store._collection_cache.add("cached_collection")
        
        result = await self.store.acollection_exists("cached_collection")
        
        assert result is True


class TestQdrantVectorStoreVectorOps:
    """Test vector operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('pyagenity.store.qdrant_vector_store.QdrantClient'), \
             patch('pyagenity.store.qdrant_vector_store.AsyncQdrantClient'):
            self.store = QdrantVectorStore(path="./test_data")
    
    def test_insert_single_vector(self, mock_qdrant_client, sample_embedding):
        """Test inserting a single vector."""
        self.store.client = mock_qdrant_client
        
        ids = self.store.insert(
            "test_collection",
            sample_embedding,
            {"content": "test"},
            "test_id"
        )
        
        assert ids == ["test_id"]
        mock_qdrant_client.upsert.assert_called_once()
        
        # Check the upsert call
        call_args = mock_qdrant_client.upsert.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        points = call_args[1]["points"]
        assert len(points) == 1
        assert points[0].id == "test_id"
        assert points[0].vector == sample_embedding
        assert points[0].payload == {"content": "test"}
    
    def test_insert_multiple_vectors(self, mock_qdrant_client):
        """Test inserting multiple vectors."""
        self.store.client = mock_qdrant_client
        
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        payloads = [{"content": "test1"}, {"content": "test2"}]
        
        ids = self.store.insert("test_collection", vectors, payloads)
        
        assert len(ids) == 2
        mock_qdrant_client.upsert.assert_called_once()
    
    def test_insert_auto_generate_ids(self, mock_qdrant_client, sample_embedding):
        """Test inserting with auto-generated IDs."""
        self.store.client = mock_qdrant_client
        
        ids = self.store.insert("test_collection", sample_embedding)
        
        assert len(ids) == 1
        assert len(ids[0]) > 0  # Should be a generated UUID
    
    @pytest.mark.asyncio
    async def test_ainsert_vector(self, mock_async_qdrant_client, sample_embedding):
        """Test async vector insertion."""
        self.store.async_client = mock_async_qdrant_client
        
        ids = await self.store.ainsert(
            "test_collection",
            sample_embedding,
            {"content": "test"},
            "test_id"
        )
        
        assert ids == ["test_id"]
        mock_async_qdrant_client.upsert.assert_called_once()
    
    def test_search_vectors(self, mock_qdrant_client, mock_point, sample_embedding):
        """Test vector search."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.search.return_value = [mock_point]
        
        results = self.store.search(
            "test_collection",
            sample_embedding,
            limit=5,
            filters={"role": "user"},
            score_threshold=0.8
        )
        
        assert len(results) == 1
        result = results[0]
        assert result.id == "test_id"
        assert result.score == 0.95
        assert result.payload == {"content": "test content", "role": "user"}
        
        mock_qdrant_client.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_asearch_vectors(self, mock_async_qdrant_client, mock_point, sample_embedding):
        """Test async vector search."""
        self.store.async_client = mock_async_qdrant_client
        mock_async_qdrant_client.search.return_value = [mock_point]
        
        results = await self.store.asearch("test_collection", sample_embedding)
        
        assert len(results) == 1
        assert results[0].id == "test_id"
    
    def test_get_vector(self, mock_qdrant_client, mock_point):
        """Test getting vector by ID."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        result = self.store.get("test_collection", "test_id")
        
        assert result is not None
        assert result.id == "test_id"
        assert result.score == 0.95
        
        mock_qdrant_client.retrieve.assert_called_once_with(
            collection_name="test_collection",
            ids=["test_id"],
            with_payload=True,
            with_vectors=True
        )
    
    def test_get_vector_not_found(self, mock_qdrant_client):
        """Test getting non-existent vector."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.retrieve.return_value = []
        
        result = self.store.get("test_collection", "nonexistent_id")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_aget_vector(self, mock_async_qdrant_client, mock_point):
        """Test async getting vector by ID."""
        self.store.async_client = mock_async_qdrant_client
        mock_async_qdrant_client.retrieve.return_value = [mock_point]
        
        result = await self.store.aget("test_collection", "test_id")
        
        assert result is not None
        assert result.id == "test_id"
    
    def test_update_vector_with_vector(self, mock_qdrant_client, sample_embedding):
        """Test updating vector with new vector data."""
        self.store.client = mock_qdrant_client
        
        self.store.update(
            "test_collection",
            "test_id",
            vector=sample_embedding,
            payload={"updated": True}
        )
        
        mock_qdrant_client.upsert.assert_called_once()
    
    def test_update_vector_payload_only(self, mock_qdrant_client):
        """Test updating only vector payload."""
        self.store.client = mock_qdrant_client
        
        self.store.update(
            "test_collection",
            "test_id",
            payload={"updated": True}
        )
        
        mock_qdrant_client.set_payload.assert_called_once_with(
            collection_name="test_collection",
            payload={"updated": True},
            points=["test_id"]
        )
    
    @pytest.mark.asyncio
    async def test_aupdate_vector(self, mock_async_qdrant_client, sample_embedding):
        """Test async vector update."""
        self.store.async_client = mock_async_qdrant_client
        
        await self.store.aupdate(
            "test_collection",
            "test_id",
            vector=sample_embedding,
            payload={"updated": True}
        )
        
        mock_async_qdrant_client.upsert.assert_called_once()
    
    def test_delete_vector(self, mock_qdrant_client):
        """Test vector deletion."""
        self.store.client = mock_qdrant_client
        
        self.store.delete("test_collection", "test_id")
        
        mock_qdrant_client.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_adelete_vector(self, mock_async_qdrant_client):
        """Test async vector deletion."""
        self.store.async_client = mock_async_qdrant_client
        
        await self.store.adelete("test_collection", "test_id")
        
        mock_async_qdrant_client.delete.assert_called_once()


class TestQdrantVectorStoreMessageOps:
    """Test message-specific operations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('pyagenity.store.qdrant_vector_store.QdrantClient'), \
             patch('pyagenity.store.qdrant_vector_store.AsyncQdrantClient'):
            self.store = QdrantVectorStore(path="./test_data")
    
    def test_store_message(self, mock_qdrant_client, sample_message, sample_embedding):
        """Test storing a message."""
        self.store.client = mock_qdrant_client
        
        # Mock embedding function
        def mock_embed_func(texts):
            return [sample_embedding]
        
        record_id = self.store.store_message(
            "messages",
            sample_message,
            mock_embed_func,
            {"conversation_id": "conv_123"}
        )
        
        assert record_id == str(sample_message.message_id)
        mock_qdrant_client.upsert.assert_called_once()
    
    def test_store_empty_message(self, mock_qdrant_client):
        """Test storing message with empty content."""
        empty_message = Message.text_message("", role="user")
        
        def mock_embed_func(texts):
            return [[0.1, 0.2]]
        
        record_id = self.store.store_message("messages", empty_message, mock_embed_func)
        
        assert record_id == ""  # Should return empty string for empty content
    
    @pytest.mark.asyncio
    async def test_astore_message(self, mock_async_qdrant_client, sample_message, sample_embedding):
        """Test async storing message."""
        self.store.async_client = mock_async_qdrant_client
        
        # Mock async embedding function
        async def mock_embed_func(texts):
            return [sample_embedding]
        
        record_id = await self.store.astore_message(
            "messages",
            sample_message,
            mock_embed_func
        )
        
        assert record_id == str(sample_message.message_id)
        mock_async_qdrant_client.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_astore_message_sync_embedding(self, mock_async_qdrant_client, sample_message, sample_embedding):
        """Test async storing message with sync embedding function."""
        self.store.async_client = mock_async_qdrant_client
        
        # Mock sync embedding function
        def mock_embed_func(texts):
            return [sample_embedding]
        
        record_id = await self.store.astore_message(
            "messages",
            sample_message,
            mock_embed_func
        )
        
        assert record_id == str(sample_message.message_id)
    
    def test_recall_similar_messages(self, mock_qdrant_client, mock_point, sample_embedding):
        """Test recalling similar messages."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.search.return_value = [mock_point]
        
        # Mock embedding function
        def mock_embed_func(texts):
            return [sample_embedding]
        
        results = self.store.recall_similar_messages(
            "messages",
            "test query",
            mock_embed_func,
            limit=3,
            score_threshold=0.7,
            role_filter="user"
        )
        
        assert len(results) == 1
        assert results[0].id == "test_id"
        
        # Verify search was called with correct filters
        call_args = mock_qdrant_client.search.call_args
        assert call_args[1]["collection_name"] == "messages"
        assert call_args[1]["query_vector"] == sample_embedding
        assert call_args[1]["limit"] == 3
        assert call_args[1]["score_threshold"] == 0.7
    
    @pytest.mark.asyncio
    async def test_arecall_similar_messages(self, mock_async_qdrant_client, mock_point, sample_embedding):
        """Test async recalling similar messages."""
        self.store.async_client = mock_async_qdrant_client
        mock_async_qdrant_client.search.return_value = [mock_point]
        
        # Mock async embedding function
        async def mock_embed_func(texts):
            return [sample_embedding]
        
        results = await self.store.arecall_similar_messages(
            "messages",
            "test query",
            mock_embed_func
        )
        
        assert len(results) == 1
        assert results[0].id == "test_id"


class TestQdrantVectorStoreUtils:
    """Test utility methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('pyagenity.store.qdrant_vector_store.QdrantClient'), \
             patch('pyagenity.store.qdrant_vector_store.AsyncQdrantClient'):
            self.store = QdrantVectorStore(path="./test_data")
    
    def test_get_collection_stats(self, mock_qdrant_client):
        """Test getting collection statistics."""
        self.store.client = mock_qdrant_client
        
        # Mock collection info
        mock_info = Mock()
        mock_info.vectors_count = 1000
        mock_info.indexed_vectors_count = 950
        mock_info.status = "green"
        mock_info.optimizer_status = "ok"
        mock_info.disk_data_size = 1024
        mock_info.ram_data_size = 512
        
        mock_qdrant_client.get_collection.return_value = mock_info
        
        stats = self.store.get_collection_stats("test_collection")
        
        assert stats["name"] == "test_collection"
        assert stats["exists"] is True
        assert stats["vectors_count"] == 1000
        assert stats["indexed_vectors_count"] == 950
        assert stats["status"] == "green"
    
    def test_get_collection_stats_error(self, mock_qdrant_client):
        """Test getting stats for non-existent collection."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")
        
        stats = self.store.get_collection_stats("nonexistent")
        
        assert stats["name"] == "nonexistent"
        assert stats["exists"] is False
        assert "error" in stats
    
    @pytest.mark.asyncio
    async def test_aget_collection_stats(self, mock_async_qdrant_client):
        """Test async getting collection statistics."""
        self.store.async_client = mock_async_qdrant_client
        
        # Mock collection info
        mock_info = Mock()
        mock_info.vectors_count = 500
        mock_info.indexed_vectors_count = 500
        mock_info.status = "green"
        mock_info.optimizer_status = "ok"
        mock_info.disk_data_size = 2048
        mock_info.ram_data_size = 1024
        
        mock_async_qdrant_client.get_collection.return_value = mock_info
        
        stats = await self.store.aget_collection_stats("test_collection")
        
        assert stats["exists"] is True
        assert stats["vectors_count"] == 500
    
    def test_cleanup(self, mock_qdrant_client):
        """Test cleanup method."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.close = Mock()
        
        self.store.cleanup()
        
        mock_qdrant_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_acleanup(self, mock_async_qdrant_client):
        """Test async cleanup method."""
        self.store.async_client = mock_async_qdrant_client
        mock_async_qdrant_client.close = AsyncMock()
        
        await self.store.acleanup()
        
        mock_async_qdrant_client.close.assert_called_once()


class TestQdrantVectorStoreHelpers:
    """Test helper methods and utilities."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('pyagenity.store.qdrant_vector_store.QdrantClient'), \
             patch('pyagenity.store.qdrant_vector_store.AsyncQdrantClient'):
            self.store = QdrantVectorStore(path="./test_data")
    
    def test_distance_metric_conversion(self):
        """Test distance metric conversion."""
        from qdrant_client.http.models import Distance
        
        assert self.store._distance_metric_to_qdrant(DistanceMetric.COSINE) == Distance.COSINE
        assert self.store._distance_metric_to_qdrant(DistanceMetric.EUCLIDEAN) == Distance.EUCLID
        assert self.store._distance_metric_to_qdrant(DistanceMetric.DOT_PRODUCT) == Distance.DOT
        assert self.store._distance_metric_to_qdrant(DistanceMetric.MANHATTAN) == Distance.MANHATTAN
    
    def test_point_to_search_result_conversion(self, mock_point):
        """Test converting Qdrant point to VectorSearchResult."""
        result = self.store._point_to_search_result(mock_point)
        
        assert isinstance(result, VectorSearchResult)
        assert result.id == "test_id"
        assert result.score == 0.95
        assert result.payload == {"content": "test content", "role": "user"}
        assert result.vector == [0.1, 0.2, 0.3]
    
    def test_build_qdrant_filter(self):
        """Test building Qdrant filters from dictionary."""
        filters = {"role": "user", "conversation_id": "conv_123", "important": True}
        
        qdrant_filter = self.store._build_qdrant_filter(filters)
        
        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 3
    
    def test_build_qdrant_filter_empty(self):
        """Test building filter with empty/None input."""
        assert self.store._build_qdrant_filter(None) is None
        assert self.store._build_qdrant_filter({}) is None


class TestQdrantVectorStoreErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Setup test fixtures."""
        with patch('pyagenity.store.qdrant_vector_store.QdrantClient'), \
             patch('pyagenity.store.qdrant_vector_store.AsyncQdrantClient'):
            self.store = QdrantVectorStore(path="./test_data")
    
    def test_insert_error_handling(self, mock_qdrant_client):
        """Test error handling during vector insertion."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.upsert.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            self.store.insert("test_collection", [0.1, 0.2, 0.3])
    
    def test_search_error_handling(self, mock_qdrant_client):
        """Test error handling during search."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.search.side_effect = Exception("Search error")
        
        with pytest.raises(Exception):
            self.store.search("test_collection", [0.1, 0.2, 0.3])
    
    def test_get_error_handling(self, mock_qdrant_client):
        """Test error handling during get operation."""
        self.store.client = mock_qdrant_client
        mock_qdrant_client.retrieve.side_effect = Exception("Retrieve error")
        
        result = self.store.get("test_collection", "test_id")
        
        assert result is None  # Should return None on error
    
    def test_invalid_embedding_function(self, sample_message):
        """Test error with invalid embedding function."""
        with pytest.raises(ValueError, match="embedding_function must be callable"):
            self.store.store_message("messages", sample_message, "not_callable")


class TestQdrantVectorStoreFactories:
    """Test factory functions."""
    
    @patch('pyagenity.store.qdrant_vector_store.QdrantVectorStore')
    def test_create_local_qdrant_vector_store(self, mock_store_cls):
        """Test local Qdrant store factory."""
        from pyagenity.store.qdrant_vector_store import create_local_qdrant_vector_store
        
        create_local_qdrant_vector_store("./custom_path")
        
        mock_store_cls.assert_called_once_with(path="./custom_path")
    
    @patch('pyagenity.store.qdrant_vector_store.QdrantVectorStore')
    def test_create_remote_qdrant_vector_store(self, mock_store_cls):
        """Test remote Qdrant store factory."""
        from pyagenity.store.qdrant_vector_store import create_remote_qdrant_vector_store
        
        create_remote_qdrant_vector_store("remote-host", 9999, "api-key")
        
        mock_store_cls.assert_called_once_with(host="remote-host", port=9999, api_key="api-key")
    
    @patch('pyagenity.store.qdrant_vector_store.QdrantVectorStore')
    def test_create_cloud_qdrant_vector_store(self, mock_store_cls):
        """Test cloud Qdrant store factory."""
        from pyagenity.store.qdrant_vector_store import create_cloud_qdrant_vector_store
        
        create_cloud_qdrant_vector_store("https://test.qdrant.io", "cloud-key")
        
        mock_store_cls.assert_called_once_with(url="https://test.qdrant.io", api_key="cloud-key")


# Integration test markers
# pytestmark = pytest.mark.asyncio  # Removed global marker to avoid warnings


class TestMemoryRecordIntegration:
    """Test MemoryRecord integration with Messages."""
    
    def test_memory_record_from_message(self, sample_message, sample_embedding):
        """Test creating MemoryRecord from Message."""
        additional_metadata = {"conversation_id": "conv_123"}
        
        record = MemoryRecord.from_message(
            sample_message,
            sample_embedding,
            additional_metadata
        )
        
        assert record.id == str(sample_message.message_id)
        assert record.content == sample_message.text()
        assert record.vector == sample_embedding
        assert record.source_type == "message"
        assert record.metadata["role"] == "user"
        assert record.metadata["conversation_id"] == "conv_123"
        assert record.metadata["message_id"] == str(sample_message.message_id)
    
    def test_memory_record_to_dict(self, sample_message, sample_embedding):
        """Test converting MemoryRecord to dictionary."""
        record = MemoryRecord.from_message(sample_message, sample_embedding)
        record_dict = record.to_dict()
        
        assert isinstance(record_dict, dict)
        assert record_dict["id"] == record.id
        assert record_dict["content"] == record.content
        assert record_dict["vector"] == sample_embedding
        assert record_dict["source_type"] == "message"
        assert isinstance(record_dict["timestamp"], str)  # ISO format


if __name__ == "__main__":
    pytest.main([__file__, "-v"])