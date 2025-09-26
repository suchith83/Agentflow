"""
Unified Base Store for PyAgenity Framework

This module provides a comprehensive base class for memory stores that can support:
- Vector-based similarity search (Qdrant, Pinecone, etc.)
- Managed memory services (mem0, Zep, etc.)
- Graph-based memory systems
- Hybrid approaches

Designed to be the single interface for all memory operations in PyAgenity.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar, Union
from uuid import uuid4

from pyagenity.utils.message import Message


# Generic type variables
DataT = TypeVar("DataT")
EmbeddingT = TypeVar("EmbeddingT", bound=Union[list[float], list[list[float]]])

logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Supported distance metrics for vector similarity."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class MemoryType(Enum):
    """Types of memories that can be stored."""

    EPISODIC = "episodic"  # Conversation memories
    SEMANTIC = "semantic"  # Facts and knowledge
    PROCEDURAL = "procedural"  # How-to knowledge
    ENTITY = "entity"  # Entity-based memories
    RELATIONSHIP = "relationship"  # Entity relationships
    CUSTOM = "custom"  # Custom memory types


@dataclass
class MemorySearchResult:
    """Result from a memory search operation."""

    id: str
    content: str
    score: float
    memory_type: str = "episodic"
    metadata: dict[str, Any] = None
    vector: list[float] = None
    user_id: str = None
    agent_id: str = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            "vector": self.vector,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class MemoryRecord:
    """Comprehensive memory record for storage."""

    content: str
    user_id: str = None
    agent_id: str = None
    memory_type: str = "episodic"
    metadata: dict[str, Any] = None
    category: str = "general"
    vector: list[float] = None
    id: str = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid4())
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

    @classmethod
    def from_message(
        cls,
        message: Message,
        user_id: str | None = None,
        agent_id: str | None = None,
        vector: list[float] | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> "MemoryRecord":
        """Create a MemoryRecord from a PyAgenity Message."""
        content = message.text()
        metadata = {
            "role": message.role,
            "message_id": str(message.message_id),
            "timestamp": message.timestamp.isoformat() if message.timestamp else None,
            "has_tool_calls": bool(message.tools_calls),
            "has_reasoning": bool(message.reasoning),
            "token_usage": message.usages.model_dump() if message.usages else None,
            **(additional_metadata or {}),
        }

        return cls(
            content=content,
            user_id=user_id,
            agent_id=agent_id,
            memory_type="episodic",
            metadata=metadata,
            vector=vector,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "memory_type": self.memory_type,
            "metadata": self.metadata,
            "category": self.category,
            "vector": self.vector,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# Type alias for embedding functions
EmbeddingFunction = Callable[[list[str]], Union[list[list[float]], list[float]]]
AsyncEmbeddingFunction = Callable[[list[str]], Union[list[list[float]], list[float]]]


class BaseStore(ABC, Generic[DataT]):
    """
    Unified base class for all memory store implementations in PyAgenity.

    This class provides a comprehensive interface that can support:
    - Vector stores (Qdrant, Pinecone, Chroma, etc.)
    - Managed memory services (mem0, Zep, etc.)
    - Graph databases (Neo4j, etc.)
    - Hybrid implementations

    Key Design Principles:
    - User and agent-centric operations
    - Support for different memory types
    - Flexible embedding integration
    - Both sync and async support
    - Extensible filtering and metadata
    - Memory lifecycle management (CRUD + search)
    """

    def __init__(
        self,
        embedding_function: EmbeddingFunction | AsyncEmbeddingFunction | None = None,
        embedding_dim: int = 768,
        **kwargs,
    ):
        """Initialize base store with optional embedding function."""
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim

    # --- Core Memory Operations (mem0-style API) ---

    @abstractmethod
    def add(
        self,
        content: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str = "episodic",
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """
        Add a new memory.

        Args:
            content: The memory content
            user_id: User identifier
            agent_id: Agent identifier
            memory_type: Type of memory (episodic, semantic, etc.)
            category: Memory category for organization
            metadata: Additional metadata
            **kwargs: Store-specific parameters

        Returns:
            Memory ID
        """

    @abstractmethod
    async def aadd(
        self,
        content: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str = "episodic",
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Async version of add."""

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        category: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """
        Search memories by content similarity.

        Args:
            query: Search query
            user_id: Filter by user
            agent_id: Filter by agent
            memory_type: Filter by memory type
            category: Filter by category
            limit: Maximum results
            score_threshold: Minimum similarity score
            filters: Additional filters
            **kwargs: Store-specific parameters

        Returns:
            List of matching memories
        """

    @abstractmethod
    async def asearch(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        category: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """Async version of search."""

    @abstractmethod
    def get(self, memory_id: str, **kwargs) -> MemorySearchResult | None:
        """Get a specific memory by ID."""

    @abstractmethod
    async def aget(self, memory_id: str, **kwargs) -> MemorySearchResult | None:
        """Async version of get."""

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Update an existing memory.

        Args:
            memory_id: ID of memory to update
            content: New content (optional)
            metadata: New/additional metadata (optional)
            **kwargs: Store-specific parameters
        """

    @abstractmethod
    async def aupdate(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Async version of update."""

    @abstractmethod
    def delete(self, memory_id: str, **kwargs) -> None:
        """Delete a memory by ID."""

    @abstractmethod
    async def adelete(self, memory_id: str, **kwargs) -> None:
        """Async version of delete."""

    # --- User/Agent Management ---

    def get_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        category: str | None = None,
        limit: int = 100,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """
        Get all memories for a user/agent with optional filters.

        This is a convenience method that uses search with empty query.
        Subclasses can override for more efficient implementations.
        """
        return self.search(
            query="",  # Empty query to get all
            user_id=user_id,
            agent_id=agent_id,
            memory_type=memory_type,
            category=category,
            limit=limit,
            **kwargs,
        )

    async def aget_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        category: str | None = None,
        limit: int = 100,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """Async version of get_all."""
        return await self.asearch(
            query="",
            user_id=user_id,
            agent_id=agent_id,
            memory_type=memory_type,
            category=category,
            limit=limit,
            **kwargs,
        )

    def delete_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        category: str | None = None,
        **kwargs,
    ) -> int:
        """
        Delete all memories matching the criteria.

        Returns:
            Number of deleted memories
        """
        # Default implementation: get all then delete individually
        # Subclasses should override for bulk deletion efficiency
        memories = self.get_all(
            user_id=user_id, agent_id=agent_id, memory_type=memory_type, category=category, **kwargs
        )

        count = 0
        for memory in memories:
            try:
                self.delete(memory.id)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete memory {memory.id}: {e}")

        return count

    async def adelete_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        category: str | None = None,
        **kwargs,
    ) -> int:
        """Async version of delete_all."""
        memories = await self.aget_all(
            user_id=user_id, agent_id=agent_id, memory_type=memory_type, category=category, **kwargs
        )

        count = 0
        for memory in memories:
            try:
                await self.adelete(memory.id)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete memory {memory.id}: {e}")

        return count

    # --- Message-Specific Convenience Methods ---

    def store_message(
        self,
        message: Message,
        user_id: str | None = None,
        agent_id: str | None = None,
        additional_metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """
        Store a PyAgenity Message as memory.

        Args:
            message: Message to store
            user_id: User identifier
            agent_id: Agent identifier
            additional_metadata: Extra metadata
            **kwargs: Store-specific parameters

        Returns:
            Memory ID
        """
        # Create memory record from message
        record = MemoryRecord.from_message(
            message, user_id, agent_id, additional_metadata=additional_metadata
        )

        # Store using main add method
        return self.add(
            content=record.content,
            user_id=record.user_id,
            agent_id=record.agent_id,
            memory_type=record.memory_type,
            metadata=record.metadata,
            **kwargs,
        )

    async def astore_message(
        self,
        message: Message,
        user_id: str | None = None,
        agent_id: str | None = None,
        additional_metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Async version of store_message."""
        record = MemoryRecord.from_message(
            message, user_id, agent_id, additional_metadata=additional_metadata
        )

        return await self.aadd(
            content=record.content,
            user_id=record.user_id,
            agent_id=record.agent_id,
            memory_type=record.memory_type,
            metadata=record.metadata,
            **kwargs,
        )

    def recall_similar_messages(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
        role_filter: str | None = None,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """
        Retrieve messages similar to the query.

        Args:
            query: Text query to find similar messages for
            user_id: Filter by user
            agent_id: Filter by agent
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            role_filter: Filter by message role
            **kwargs: Store-specific parameters

        Returns:
            List of similar message records
        """
        # Build filters
        filters = {}
        if role_filter:
            filters["role"] = role_filter

        return self.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            memory_type="episodic",  # Messages are episodic memories
            limit=limit,
            score_threshold=score_threshold,
            filters=filters if filters else None,
            **kwargs,
        )

    async def arecall_similar_messages(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
        role_filter: str | None = None,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """Async version of recall_similar_messages."""
        filters = {}
        if role_filter:
            filters["role"] = role_filter

        return await self.asearch(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            memory_type="episodic",
            limit=limit,
            score_threshold=score_threshold,
            filters=filters if filters else None,
            **kwargs,
        )

    # --- Embedding Utilities ---

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using configured embedding function."""
        if not self.embedding_function:
            raise ValueError("No embedding function configured")

        if callable(self.embedding_function):
            result = self.embedding_function([text])
            # Handle different return formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result[0]  # List of embeddings, take first
                return result  # Single embedding
            raise ValueError("Embedding function returned unexpected format")

        raise ValueError("Embedding function is not callable")

    async def _agenerate_embedding(self, text: str) -> list[float]:
        """Async version of _generate_embedding."""
        if not self.embedding_function:
            raise ValueError("No embedding function configured")

        if asyncio.iscoroutinefunction(self.embedding_function):
            result = await self.embedding_function([text])
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result[0]
                return result
            raise ValueError("Embedding function returned unexpected format")

        # Fallback to sync version
        return self._generate_embedding(text)

    # --- Statistics and Management ---

    def get_stats(self, user_id: str | None = None, agent_id: str | None = None) -> dict[str, Any]:
        """
        Get statistics about stored memories.

        Args:
            user_id: Filter stats by user
            agent_id: Filter stats by agent

        Returns:
            Dictionary with statistics
        """
        # Default implementation provides basic stats
        # Subclasses should override for detailed statistics
        try:
            memories = self.get_all(user_id=user_id, agent_id=agent_id, limit=1000)
            return {
                "total_memories": len(memories),
                "user_id": user_id,
                "agent_id": agent_id,
                "memory_types": list({m.memory_type for m in memories}),
                "categories": list({m.metadata.get("category", "general") for m in memories}),
            }
        except Exception as e:
            return {"error": str(e)}

    async def aget_stats(
        self, user_id: str | None = None, agent_id: str | None = None
    ) -> dict[str, Any]:
        """Async version of get_stats."""
        try:
            memories = await self.aget_all(user_id=user_id, agent_id=agent_id, limit=1000)
            return {
                "total_memories": len(memories),
                "user_id": user_id,
                "agent_id": agent_id,
                "memory_types": list({m.memory_type for m in memories}),
                "categories": list({m.metadata.get("category", "general") for m in memories}),
            }
        except Exception as e:
            return {"error": str(e)}

    # --- Cleanup and Resource Management ---

    def cleanup(self) -> None:
        """Clean up any resources used by the store."""
        # Default implementation does nothing

    async def acleanup(self) -> None:
        """Async version of cleanup."""

    # --- Vector Store Compatibility (Optional) ---

    # These methods provide compatibility with vector store implementations
    # Subclasses can choose to implement these for vector-specific operations

    def create_collection(
        self,
        name: str,
        vector_size: int | None = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs,
    ) -> None:
        """Create a collection/namespace (optional for vector stores)."""
        # Many stores don't need explicit collection creation

    async def acreate_collection(
        self,
        name: str,
        vector_size: int | None = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs,
    ) -> None:
        """Async version of create_collection."""

    def collection_exists(self, name: str) -> bool:
        """Check if collection exists (optional)."""
        return True  # Default: assume collections exist

    async def acollection_exists(self, name: str) -> bool:
        """Async version of collection_exists."""
        return True

    def list_collections(self) -> list[str]:
        """List all collections (optional)."""
        return []  # Default: no collections

    async def alist_collections(self) -> list[str]:
        """Async version of list_collections."""
        return []

    def delete_collection(self, name: str) -> None:
        """Delete a collection (optional)."""

    async def adelete_collection(self, name: str) -> None:
        """Async version of delete_collection."""

    # --- Legacy Support ---

    # These methods provide backward compatibility with the old base store interface
    # They map to the new unified API

    def update_memory(
        self,
        config: dict[str, Any],
        info: DataT,
    ) -> None:
        """Legacy method - maps to add/update operations."""
        # This is a compatibility shim - subclasses should override if needed
        if hasattr(info, "content"):
            content = getattr(info, "content", str(info))
            user_id = config.get("user_id")
            agent_id = config.get("agent_id")
            metadata = config.get("metadata", {})

            # Try to update first, then add if not exists
            memory_id = config.get("memory_id")
            if memory_id:
                try:
                    self.update(memory_id, content=content, metadata=metadata)
                except Exception:
                    # If update fails, add new memory
                    self.add(content, user_id=user_id, agent_id=agent_id, metadata=metadata)
            else:
                self.add(content, user_id=user_id, agent_id=agent_id, metadata=metadata)
        else:
            raise NotImplementedError("update_memory requires content-aware info object")

    async def aupdate_memory(
        self,
        config: dict[str, Any],
        info: DataT,
    ) -> None:
        """Legacy async method - maps to add/update operations."""
        if hasattr(info, "content"):
            content = getattr(info, "content", str(info))
            user_id = config.get("user_id")
            agent_id = config.get("agent_id")
            metadata = config.get("metadata", {})

            memory_id = config.get("memory_id")
            if memory_id:
                try:
                    await self.aupdate(memory_id, content=content, metadata=metadata)
                except Exception:
                    await self.aadd(content, user_id=user_id, agent_id=agent_id, metadata=metadata)
            else:
                await self.aadd(content, user_id=user_id, agent_id=agent_id, metadata=metadata)
        else:
            raise NotImplementedError("aupdate_memory requires content-aware info object")

    def get_memory(
        self,
        config: dict[str, Any],
    ) -> DataT | None:
        """Legacy method - maps to get operation."""
        memory_id = config.get("memory_id")
        if memory_id:
            result = self.get(memory_id)
            return result if result else None

        # If no ID provided, search based on config
        user_id = config.get("user_id")
        agent_id = config.get("agent_id")
        query = config.get("query", "")
        results = self.search(query, user_id=user_id, agent_id=agent_id, limit=1)
        return results[0] if results else None

    async def aget_memory(
        self,
        config: dict[str, Any],
    ) -> DataT | None:
        """Legacy async method - maps to get operation."""
        memory_id = config.get("memory_id")
        if memory_id:
            result = await self.aget(memory_id)
            return result if result else None

        user_id = config.get("user_id")
        agent_id = config.get("agent_id")
        query = config.get("query", "")
        results = await self.asearch(query, user_id=user_id, agent_id=agent_id, limit=1)
        return results[0] if results else None

    def delete_memory(
        self,
        config: dict[str, Any],
    ) -> None:
        """Legacy method - maps to delete operation."""
        memory_id = config.get("memory_id")
        if memory_id:
            self.delete(memory_id)
        else:
            # Delete by criteria
            user_id = config.get("user_id")
            agent_id = config.get("agent_id")
            self.delete_all(user_id=user_id, agent_id=agent_id)

    async def adelete_memory(
        self,
        config: dict[str, Any],
    ) -> None:
        """Legacy async method - maps to delete operation."""
        memory_id = config.get("memory_id")
        if memory_id:
            await self.adelete(memory_id)
        else:
            user_id = config.get("user_id")
            agent_id = config.get("agent_id")
            await self.adelete_all(user_id=user_id, agent_id=agent_id)

    def related_memory(
        self,
        config: dict[str, Any],
        query: str,
    ) -> list[DataT]:
        """Legacy method - maps to search operation."""
        user_id = config.get("user_id")
        agent_id = config.get("agent_id")
        limit = config.get("limit", 10)
        return self.search(query, user_id=user_id, agent_id=agent_id, limit=limit)

    async def arelated_memory(
        self,
        config: dict[str, Any],
        query: str,
    ) -> list[DataT]:
        """Legacy async method - maps to search operation."""
        user_id = config.get("user_id")
        agent_id = config.get("agent_id")
        limit = config.get("limit", 10)
        return await self.asearch(query, user_id=user_id, agent_id=agent_id, limit=limit)

    def release(self) -> None:
        """Legacy method - maps to cleanup."""
        self.cleanup()

    async def arelease(self) -> None:
        """Legacy async method - maps to cleanup."""
        await self.acleanup()


# Convenience type aliases
MessageMemoryStore = BaseStore[Message]
VectorStoreBase = BaseStore  # For backward compatibility
