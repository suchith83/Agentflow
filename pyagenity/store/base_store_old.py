"""
Simplified Async-First Base Store for PyAgenity Framework

This module provides a clean, modern interface for memory stores with:
- Async-first design for better performance
- Core CRUD operations only
- Message-specific convenience methods
- Extensible for different backends (vector stores, managed services, etc.)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, TypeVar, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

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


class MemorySearchResult(BaseModel):
    """Result from a memory search operation (Pydantic model)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(default="", description="Primary textual content of the memory")
    score: float = Field(default=0.0, ge=0.0, description="Similarity / relevance score")
    memory_type: str = Field(default="episodic")
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = Field(default=None)
    user_id: str | None = None
    agent_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # Backward-compatible alias for payload if vector-store returns payload field
    PAYLOAD_KEYS: ClassVar[set[str]] = {"payload", "metadata"}

    @field_validator("memory_type")
    @classmethod
    def normalize_memory_type(cls, v: str) -> str:
        return v or "episodic"

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v):
        if v is not None and (
            not isinstance(v, list) or any(not isinstance(x, (int, float)) for x in v)
        ):
            raise ValueError("vector must be list[float] or None")
        return v

    @model_validator(mode="after")
    def ensure_timestamps(self):
        now = datetime.now()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
        return self

    def touch(self):
        """Update updated_at timestamp."""
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        # Serialize datetimes
        if data.get("created_at"):
            data["created_at"] = data["created_at"].isoformat()
        if data.get("updated_at"):
            data["updated_at"] = data["updated_at"].isoformat()
        return data


class MemoryRecord(BaseModel):
    """Comprehensive memory record for storage (Pydantic model)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    user_id: str | None = None
    agent_id: str | None = None
    memory_type: str = Field(default="episodic")
    metadata: dict[str, Any] = Field(default_factory=dict)
    category: str = Field(default="general")
    vector: list[float] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, v):
        if v is not None and (
            not isinstance(v, list) or any(not isinstance(x, (int, float)) for x in v)
        ):
            raise ValueError("vector must be list[float] or None")
        return v

    @model_validator(mode="after")
    def set_timestamps(self):
        now = datetime.now()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
        return self

    @classmethod
    def from_message(
        cls,
        message: Message,
        user_id: str | None = None,
        agent_id: str | None = None,
        vector: list[float] | None = None,
        additional_metadata: dict[str, Any] | None = None,
    ) -> "MemoryRecord":
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
        data = self.model_dump()
        if data.get("created_at"):
            data["created_at"] = data["created_at"].isoformat()
        if data.get("updated_at"):
            data["updated_at"] = data["updated_at"].isoformat()
        return data


# Type alias for embedding functions
EmbeddingFunction = Callable[[list[str]], Union[list[list[float]], list[float]]]


class BaseStore(ABC):
    """
    Simplified async-first base class for memory stores in PyAgenity.

    This class provides a clean interface that supports:
    - Vector stores (Qdrant, Pinecone, Chroma, etc.)
    - Managed memory services (mem0, Zep, etc.)
    - Graph databases (Neo4j, etc.)

    Key Design Principles:
    - Async-first for better performance
    - Core CRUD operations only
    - User and agent-centric operations
    - Extensible filtering and metadata
    """

    def __init__(
        self,
        embedding_function: EmbeddingFunction | None = None,
        embedding_dim: int = 768,
        **kwargs,
    ):
        """Initialize base store with optional embedding function."""
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim

    # --- Core Memory Operations ---

    @abstractmethod
    async def add(
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
    async def search(
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
    async def get(self, memory_id: str, **kwargs) -> MemorySearchResult | None:
        """Get a specific memory by ID."""

    @abstractmethod
    async def update(
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
    async def delete(self, memory_id: str, **kwargs) -> None:
        """Delete a memory by ID."""

    # --- Message-Specific Convenience Methods ---

    async def store_message(
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
        return await self.add(
            content=record.content,
            user_id=record.user_id,
            agent_id=record.agent_id,
            memory_type=record.memory_type,
            metadata=record.metadata,
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
                # Single embedding list[float]
                return result  # type: ignore[return-value]
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
                return result  # type: ignore[return-value]
            raise ValueError("Embedding function returned unexpected format")

        # Fallback to sync version
        return self._generate_embedding(text)

    # --- Statistics and Management ---

    async def get_stats(
        self, user_id: str | None = None, agent_id: str | None = None
    ) -> dict[str, Any]:
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
            memories = await self.search("", user_id=user_id, agent_id=agent_id, limit=1000)
            return {
                "total_memories": len(memories),
                "user_id": user_id,
                "agent_id": agent_id,
                "memory_types": list({m.memory_type for m in memories}),
                "categories": list({m.metadata.get("category", "general") for m in memories}),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_memories": 0, "user_id": user_id, "agent_id": agent_id, "error": str(e)}

    # --- Cleanup and Resource Management ---

    async def cleanup(self) -> None:
        """Clean up any resources used by the store (override in subclasses if needed)."""
        logger.debug("BaseStore.cleanup called - no action by default")

    # --- Sync Wrappers for Convenience ---

    def add_sync(self, *args, **kwargs) -> str:
        """Sync wrapper for add method."""
        return asyncio.run(self.add(*args, **kwargs))

    def search_sync(self, *args, **kwargs) -> list[MemorySearchResult]:
        """Sync wrapper for search method."""
        return asyncio.run(self.search(*args, **kwargs))

    def get_sync(self, *args, **kwargs) -> MemorySearchResult | None:
        """Sync wrapper for get method."""
        return asyncio.run(self.get(*args, **kwargs))

    def update_sync(self, *args, **kwargs) -> None:
        """Sync wrapper for update method."""
        return asyncio.run(self.update(*args, **kwargs))

    def delete_sync(self, *args, **kwargs) -> None:
        """Sync wrapper for delete method."""
        return asyncio.run(self.delete(*args, **kwargs))

    def store_message_sync(self, *args, **kwargs) -> str:
        """Sync wrapper for store_message method."""
        return asyncio.run(self.store_message(*args, **kwargs))

    def get_stats_sync(self, *args, **kwargs) -> dict[str, Any]:
        """Sync wrapper for get_stats method."""
        return asyncio.run(self.get_stats(*args, **kwargs))

    def cleanup_sync(self) -> None:
        """Sync wrapper for cleanup method."""
        return asyncio.run(self.cleanup())


# Convenience type aliases for backward compatibility
MessageMemoryStore = BaseStore  # Simplified alias
VectorStoreBase = BaseStore  # For backward compatibility