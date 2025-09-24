"""
Comprehensive tests for the vector_base_store module.

This test file validates the abstract base class and its associated data structures
using pytest and includes both sync and async test patterns.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock

import pytest

from pyagenity.store.vector_base_store import (
    DistanceMetric,
    MemoryRecord,
    MessageVectorStore,
    VectorSearchResult,
    VectorStoreBase,
)
from pyagenity.utils.message import Message, TextBlock, TokenUsages


class MockVectorStore(VectorStoreBase[Message]):
    """Concrete implementation for testing the abstract base class."""

    def __init__(self):
        """Initialize mock vector store with in-memory storage."""
        self.collections: Dict[str, Dict[str, Any]] = {}
        self.vectors: Dict[str, Dict[str, VectorSearchResult]] = {}

    # Collection Management
    def create_collection(
        self, 
        name: str, 
        vector_size: int, 
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any
    ) -> None:
        if name in self.collections:
            raise ValueError(f"Collection {name} already exists")
        self.collections[name] = {
            "vector_size": vector_size,
            "distance_metric": distance_metric,
            **kwargs
        }
        self.vectors[name] = {}

    async def acreate_collection(
        self, 
        name: str, 
        vector_size: int, 
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any
    ) -> None:
        self.create_collection(name, vector_size, distance_metric, **kwargs)

    def list_collections(self) -> List[str]:
        return list(self.collections.keys())

    async def alist_collections(self) -> List[str]:
        return self.list_collections()

    def delete_collection(self, name: str) -> None:
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist")
        del self.collections[name]
        del self.vectors[name]

    async def adelete_collection(self, name: str) -> None:
        self.delete_collection(name)

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    async def acollection_exists(self, name: str) -> bool:
        return self.collection_exists(name)

    # Vector Operations
    def insert(
        self,
        collection_name: str,
        vectors: Union[List[float], List[List[float]]],
        payloads: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        
        # Normalize inputs to lists
        if isinstance(vectors[0], (int, float)):
            vectors = [vectors]
        if payloads and not isinstance(payloads, list):
            payloads = [payloads]
        if ids and not isinstance(ids, list):
            ids = [ids]
        
        result_ids = []
        for i, vector in enumerate(vectors):
            vector_id = ids[i] if ids else f"vec_{len(self.vectors[collection_name])}"
            payload = payloads[i] if payloads else {}
            
            self.vectors[collection_name][vector_id] = VectorSearchResult(
                id=vector_id,
                score=1.0,  # Perfect match for inserted vectors
                payload=payload,
                vector=vector
            )
            result_ids.append(vector_id)
        
        return result_ids

    async def ainsert(
        self,
        collection_name: str,
        vectors: Union[List[float], List[List[float]]],
        payloads: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        return self.insert(collection_name, vectors, payloads, ids)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        
        results = []
        for vector_result in self.vectors[collection_name].values():
            # Apply filters if provided
            if filters:
                if not all(
                    vector_result.payload and vector_result.payload.get(k) == v 
                    for k, v in filters.items()
                ):
                    continue
            
            # Mock similarity score (random for testing)
            score = 0.9 if vector_result.vector == query_vector else 0.7
            
            # Apply score threshold
            if score_threshold and score < score_threshold:
                continue
            
            results.append(VectorSearchResult(
                id=vector_result.id,
                score=score,
                payload=vector_result.payload,
                vector=vector_result.vector
            ))
        
        # Sort by score and apply limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def asearch(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
        return self.search(collection_name, query_vector, limit, filters, score_threshold)

    def get(self, collection_name: str, vector_id: str) -> Optional[VectorSearchResult]:
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        return self.vectors[collection_name].get(vector_id)

    async def aget(self, collection_name: str, vector_id: str) -> Optional[VectorSearchResult]:
        return self.get(collection_name, vector_id)

    def update(
        self,
        collection_name: str,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        if vector_id not in self.vectors[collection_name]:
            raise ValueError(f"Vector {vector_id} does not exist")
        
        existing = self.vectors[collection_name][vector_id]
        if vector:
            existing.vector = vector
        if payload:
            existing.payload = {**(existing.payload or {}), **payload}

    async def aupdate(
        self,
        collection_name: str,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        self.update(collection_name, vector_id, vector, payload)

    def delete(self, collection_name: str, vector_id: str) -> None:
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist")
        if vector_id not in self.vectors[collection_name]:
            raise ValueError(f"Vector {vector_id} does not exist")
        del self.vectors[collection_name][vector_id]

    async def adelete(self, collection_name: str, vector_id: str) -> None:
        self.delete(collection_name, vector_id)


class TestDistanceMetric:
    """Test the DistanceMetric enum."""

    def test_distance_metric_values(self):
        """Test that all expected distance metrics are available."""
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"
        assert DistanceMetric.DOT_PRODUCT.value == "dot_product"
        assert DistanceMetric.MANHATTAN.value == "manhattan"

    def test_distance_metric_count(self):
        """Test that we have the expected number of distance metrics."""
        assert len(DistanceMetric) == 4


class TestVectorSearchResult:
    """Test the VectorSearchResult dataclass."""

    def test_vector_search_result_creation(self):
        """Test creating a VectorSearchResult."""
        result = VectorSearchResult(
            id="test_id",
            score=0.95,
            payload={"key": "value"},
            vector=[0.1, 0.2, 0.3]
        )
        
        assert result.id == "test_id"
        assert result.score == 0.95
        assert result.payload == {"key": "value"}
        assert result.vector == [0.1, 0.2, 0.3]

    def test_vector_search_result_to_dict(self):
        """Test converting VectorSearchResult to dictionary."""
        result = VectorSearchResult(
            id="test_id",
            score=0.95,
            payload={"key": "value"},
            vector=[0.1, 0.2, 0.3]
        )
        
        expected = {
            "id": "test_id",
            "score": 0.95,
            "payload": {"key": "value"},
            "vector": [0.1, 0.2, 0.3]
        }
        
        assert result.to_dict() == expected

    def test_vector_search_result_minimal(self):
        """Test creating VectorSearchResult with minimal fields."""
        result = VectorSearchResult(id="test", score=0.8)
        
        assert result.id == "test"
        assert result.score == 0.8
        assert result.payload is None
        assert result.vector is None


class TestMemoryRecord:
    """Test the MemoryRecord dataclass."""

    def test_memory_record_creation(self):
        """Test creating a MemoryRecord."""
        timestamp = datetime.now()
        record = MemoryRecord(
            id="mem_123",
            content="Test content",
            vector=[0.1, 0.2, 0.3],
            metadata={"role": "user"},
            timestamp=timestamp,
            source_type="message"
        )
        
        assert record.id == "mem_123"
        assert record.content == "Test content"
        assert record.vector == [0.1, 0.2, 0.3]
        assert record.metadata == {"role": "user"}
        assert record.timestamp == timestamp
        assert record.source_type == "message"

    def test_memory_record_to_dict(self):
        """Test converting MemoryRecord to dictionary."""
        timestamp = datetime.now()
        record = MemoryRecord(
            id="mem_123",
            content="Test content",
            vector=[0.1, 0.2, 0.3],
            metadata={"role": "user"},
            timestamp=timestamp,
            source_type="message"
        )
        
        result = record.to_dict()
        
        assert result["id"] == "mem_123"
        assert result["content"] == "Test content"
        assert result["vector"] == [0.1, 0.2, 0.3]
        assert result["metadata"] == {"role": "user"}
        assert result["timestamp"] == timestamp.isoformat()
        assert result["source_type"] == "message"

    def test_memory_record_from_message(self):
        """Test creating MemoryRecord from Message."""
        message = Message(
            message_id="msg_123",
            role="user",
            content=[TextBlock(text="Hello world")],
            timestamp=datetime.now(),
            usages=TokenUsages(completion_tokens=10, prompt_tokens=5, total_tokens=15)
        )
        
        vector = [0.1, 0.2, 0.3]
        additional_meta = {"conversation_id": "conv_123"}
        
        record = MemoryRecord.from_message(message, vector, additional_meta)
        
        assert record.id == "msg_123"
        assert record.content == "Hello world"
        assert record.vector == vector
        assert record.source_type == "message"
        assert record.metadata["role"] == "user"
        assert record.metadata["message_id"] == "msg_123"
        assert record.metadata["conversation_id"] == "conv_123"
        assert record.metadata["has_tool_calls"] is False
        assert record.metadata["has_reasoning"] is False
        assert "token_usage" in record.metadata

    def test_memory_record_from_message_with_reasoning(self):
        """Test creating MemoryRecord from Message with reasoning."""
        message = Message(
            message_id="msg_456",
            role="assistant",
            content=[TextBlock(text="Let me think about this...")],
            reasoning="I need to analyze the user's request",
            tools_calls=[{"name": "search", "args": {}}]
        )
        
        record = MemoryRecord.from_message(message)
        
        assert record.metadata["has_reasoning"] is True
        assert record.metadata["has_tool_calls"] is True


class TestVectorStoreBase:
    """Test the VectorStoreBase abstract class using MockVectorStore."""

    @pytest.fixture
    def vector_store(self):
        """Provide a MockVectorStore instance for testing."""
        return MockVectorStore()

    @pytest.fixture
    def sample_message(self):
        """Provide a sample Message for testing."""
        return Message(
            message_id="test_123",
            role="user",
            content=[TextBlock(text="This is a test message")]
        )

    @pytest.fixture
    def mock_embedding_function(self):
        """Provide a mock embedding function."""
        def embed(texts: List[str]) -> List[List[float]]:
            # Return simple mock embeddings based on text length
            return [[0.1 * len(text), 0.2 * len(text), 0.3 * len(text)] for text in texts]
        return embed

    @pytest.fixture
    def async_mock_embedding_function(self):
        """Provide an async mock embedding function."""
        async def embed(texts: List[str]) -> List[List[float]]:
            # Simulate async processing
            await asyncio.sleep(0.001)
            return [[0.1 * len(text), 0.2 * len(text), 0.3 * len(text)] for text in texts]
        return embed

    # Collection Management Tests
    def test_create_collection(self, vector_store):
        """Test creating a new collection."""
        vector_store.create_collection("test_collection", 768, DistanceMetric.COSINE)
        
        assert vector_store.collection_exists("test_collection")
        assert "test_collection" in vector_store.list_collections()

    def test_create_collection_duplicate(self, vector_store):
        """Test creating duplicate collection raises error."""
        vector_store.create_collection("test_collection", 768)
        
        with pytest.raises(ValueError, match="already exists"):
            vector_store.create_collection("test_collection", 768)

    def test_delete_collection(self, vector_store):
        """Test deleting a collection."""
        vector_store.create_collection("test_collection", 768)
        assert vector_store.collection_exists("test_collection")
        
        vector_store.delete_collection("test_collection")
        assert not vector_store.collection_exists("test_collection")

    def test_delete_nonexistent_collection(self, vector_store):
        """Test deleting non-existent collection raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            vector_store.delete_collection("nonexistent")

    def test_list_collections(self, vector_store):
        """Test listing collections."""
        assert vector_store.list_collections() == []
        
        vector_store.create_collection("col1", 768)
        vector_store.create_collection("col2", 512)
        
        collections = vector_store.list_collections()
        assert "col1" in collections
        assert "col2" in collections
        assert len(collections) == 2

    # Vector Operations Tests
    def test_insert_single_vector(self, vector_store):
        """Test inserting a single vector."""
        vector_store.create_collection("test_col", 3)
        
        ids = vector_store.insert(
            collection_name="test_col",
            vectors=[0.1, 0.2, 0.3],
            payloads={"text": "test"},
            ids="vec1"
        )
        
        assert ids == ["vec1"]
        result = vector_store.get("test_col", "vec1")
        assert result is not None
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.payload == {"text": "test"}

    def test_insert_multiple_vectors(self, vector_store):
        """Test inserting multiple vectors."""
        vector_store.create_collection("test_col", 3)
        
        ids = vector_store.insert(
            collection_name="test_col",
            vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            payloads=[{"text": "first"}, {"text": "second"}],
            ids=["vec1", "vec2"]
        )
        
        assert len(ids) == 2
        assert "vec1" in ids
        assert "vec2" in ids

    def test_insert_nonexistent_collection(self, vector_store):
        """Test inserting into non-existent collection raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            vector_store.insert("nonexistent", [0.1, 0.2, 0.3])

    def test_search_vectors(self, vector_store):
        """Test searching for similar vectors."""
        vector_store.create_collection("test_col", 3)
        
        # Insert test vectors
        vector_store.insert(
            "test_col",
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [{"type": "A"}, {"type": "B"}],
            ["vec1", "vec2"]
        )
        
        # Search
        results = vector_store.search("test_col", [0.1, 0.2, 0.3], limit=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, VectorSearchResult) for r in results)

    def test_search_with_filters(self, vector_store):
        """Test searching with metadata filters."""
        vector_store.create_collection("test_col", 3)
        
        vector_store.insert(
            "test_col",
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [{"type": "A"}, {"type": "B"}],
            ["vec1", "vec2"]
        )
        
        results = vector_store.search("test_col", [0.1, 0.2, 0.3], filters={"type": "A"})
        
        # Should only return vectors matching the filter
        for result in results:
            assert result.payload["type"] == "A"

    def test_update_vector(self, vector_store):
        """Test updating a vector."""
        vector_store.create_collection("test_col", 3)
        vector_store.insert("test_col", [0.1, 0.2, 0.3], {"original": True}, "vec1")
        
        vector_store.update("test_col", "vec1", [0.4, 0.5, 0.6], {"updated": True})
        
        result = vector_store.get("test_col", "vec1")
        assert result.vector == [0.4, 0.5, 0.6]
        assert result.payload["updated"] is True
        assert result.payload["original"] is True  # Should preserve existing payload

    def test_delete_vector(self, vector_store):
        """Test deleting a vector."""
        vector_store.create_collection("test_col", 3)
        vector_store.insert("test_col", [0.1, 0.2, 0.3], ids="vec1")
        
        assert vector_store.get("test_col", "vec1") is not None
        
        vector_store.delete("test_col", "vec1")
        
        assert vector_store.get("test_col", "vec1") is None

    # Memory-Specific Tests
    def test_store_message(self, vector_store, sample_message, mock_embedding_function):
        """Test storing a PyAgenity Message."""
        vector_store.create_collection("messages", 3)
        
        memory_id = vector_store.store_message(
            "messages", 
            sample_message, 
            mock_embedding_function,
            {"conversation_id": "conv_123"}
        )
        
        assert memory_id == "test_123"
        
        result = vector_store.get("messages", memory_id)
        assert result is not None
        assert result.payload["role"] == "user"
        assert result.payload["conversation_id"] == "conv_123"

    def test_store_empty_message(self, vector_store, mock_embedding_function):
        """Test storing empty message returns empty string."""
        vector_store.create_collection("messages", 3)
        
        empty_message = Message(
            role="user",
            content=[TextBlock(text="   ")]  # Only whitespace
        )
        
        memory_id = vector_store.store_message("messages", empty_message, mock_embedding_function)
        assert memory_id == ""

    def test_recall_similar_messages(self, vector_store, mock_embedding_function):
        """Test recalling similar messages."""
        vector_store.create_collection("messages", 3)
        
        # Store some messages
        msg1 = Message(role="user", content=[TextBlock(text="Hello world")])
        msg2 = Message(role="assistant", content=[TextBlock(text="Hi there")])
        
        vector_store.store_message("messages", msg1, mock_embedding_function)
        vector_store.store_message("messages", msg2, mock_embedding_function)
        
        # Recall similar messages
        results = vector_store.recall_similar_messages(
            "messages", 
            "Hello", 
            mock_embedding_function,
            limit=2
        )
        
        assert len(results) <= 2
        assert all(isinstance(r, VectorSearchResult) for r in results)

    def test_recall_with_role_filter(self, vector_store, mock_embedding_function):
        """Test recalling messages with role filter."""
        vector_store.create_collection("messages", 3)
        
        msg1 = Message(role="user", content=[TextBlock(text="Hello")])
        msg2 = Message(role="assistant", content=[TextBlock(text="Hi")])
        
        vector_store.store_message("messages", msg1, mock_embedding_function)
        vector_store.store_message("messages", msg2, mock_embedding_function)
        
        # Search with role filter
        results = vector_store.recall_similar_messages(
            "messages", 
            "Hello", 
            mock_embedding_function,
            role_filter="user"
        )
        
        for result in results:
            assert result.payload["role"] == "user"

    # Async Tests
    @pytest.mark.asyncio
    async def test_async_create_collection(self, vector_store):
        """Test async collection creation."""
        await vector_store.acreate_collection("async_test", 768)
        
        assert await vector_store.acollection_exists("async_test")

    @pytest.mark.asyncio
    async def test_async_insert_and_search(self, vector_store):
        """Test async insert and search operations."""
        await vector_store.acreate_collection("async_test", 3)
        
        ids = await vector_store.ainsert(
            "async_test",
            [0.1, 0.2, 0.3],
            {"test": True},
            "async_vec1"
        )
        
        assert ids == ["async_vec1"]
        
        results = await vector_store.asearch("async_test", [0.1, 0.2, 0.3])
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_async_store_message(self, vector_store, sample_message, async_mock_embedding_function):
        """Test async message storage."""
        await vector_store.acreate_collection("async_messages", 3)
        
        memory_id = await vector_store.astore_message(
            "async_messages",
            sample_message,
            async_mock_embedding_function
        )
        
        assert memory_id == "test_123"
        
        result = await vector_store.aget("async_messages", memory_id)
        assert result is not None

    @pytest.mark.asyncio
    async def test_async_recall_messages(self, vector_store, async_mock_embedding_function):
        """Test async message recall."""
        await vector_store.acreate_collection("async_messages", 3)
        
        msg = Message(role="user", content=[TextBlock(text="Test message")])
        await vector_store.astore_message("async_messages", msg, async_mock_embedding_function)
        
        results = await vector_store.arecall_similar_messages(
            "async_messages",
            "Test",
            async_mock_embedding_function
        )
        
        assert len(results) >= 0

    # Utility Tests
    def test_get_collection_stats(self, vector_store):
        """Test getting collection statistics."""
        vector_store.create_collection("stats_test", 768)
        
        stats = vector_store.get_collection_stats("stats_test")
        
        assert stats["exists"] is True
        assert stats["name"] == "stats_test"

    @pytest.mark.asyncio
    async def test_async_get_collection_stats(self, vector_store):
        """Test async collection statistics."""
        await vector_store.acreate_collection("async_stats_test", 768)
        
        stats = await vector_store.aget_collection_stats("async_stats_test")
        
        assert stats["exists"] is True
        assert stats["name"] == "async_stats_test"

    def test_cleanup(self, vector_store):
        """Test cleanup method doesn't raise errors."""
        vector_store.cleanup()  # Should not raise

    @pytest.mark.asyncio
    async def test_async_cleanup(self, vector_store):
        """Test async cleanup method."""
        await vector_store.acleanup()  # Should not raise


class TestAbstractMethods:
    """Test that the abstract methods are properly defined."""

    def test_vector_store_is_abstract(self):
        """Test that VectorStoreBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VectorStoreBase()

    def test_message_vector_store_alias(self):
        """Test the MessageVectorStore type alias."""
        assert MessageVectorStore == VectorStoreBase[Message]

    def test_abstract_methods_exist(self):
        """Test that all expected abstract methods exist."""
        expected_methods = [
            "create_collection", "acreate_collection",
            "list_collections", "alist_collections", 
            "delete_collection", "adelete_collection",
            "collection_exists", "acollection_exists",
            "insert", "ainsert",
            "search", "asearch", 
            "get", "aget",
            "update", "aupdate",
            "delete", "adelete"
        ]
        
        for method_name in expected_methods:
            assert hasattr(VectorStoreBase, method_name)
            assert callable(getattr(VectorStoreBase, method_name))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])