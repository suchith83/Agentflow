"""
Simplified Qdrant Vector Store Implementation for PyAgenity Framework

This module provides a clean, async-first implementation of BaseStore using Qdrant
as the backend vector database. Supports both local and cloud Qdrant deployments.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from .base_store import BaseStore, DistanceMetric, MemorySearchResult


try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except ImportError:
    raise ImportError("Qdrant client not installed. Install with: pip install qdrant-client")

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseStore):
    """
    Simplified async-first Qdrant-based vector store implementation.

    Features:
    - Async-only operations for better performance
    - Local and cloud Qdrant deployment support
    - Efficient vector similarity search
    - Collection management with automatic creation
    - Rich metadata filtering capabilities

    Example:
        ```python
        # Local Qdrant
        store = QdrantVectorStore(path="./qdrant_data")

        # Remote Qdrant
        store = QdrantVectorStore(host="localhost", port=6333)

        # Cloud Qdrant
        store = QdrantVectorStore(url="https://xyz.qdrant.io", api_key="your-api-key")
        ```
    """

    def __init__(
        self,
        path: str | None = None,
        host: str | None = None,
        port: int | None = None,
        url: str | None = None,
        api_key: str | None = None,
        default_collection: str = "pyagenity_vectors",
        vector_size: int | None = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            path: Path for local Qdrant (file-based storage)
            host: Host for remote Qdrant server
            port: Port for remote Qdrant server
            url: URL for Qdrant cloud
            api_key: API key for Qdrant cloud
            default_collection: Default collection name
            vector_size: Default vector size for auto-collection creation
            distance_metric: Default distance metric
            **kwargs: Additional client parameters
        """
        super().__init__(**kwargs)

        # Initialize async client
        if path:
            self.client = AsyncQdrantClient(path=path, **kwargs)
        elif url:
            self.client = AsyncQdrantClient(url=url, api_key=api_key, **kwargs)
        else:
            host = host or "localhost"
            port = port or 6333
            self.client = AsyncQdrantClient(host=host, port=port, api_key=api_key, **kwargs)

        # Cache for collection existence checks
        self._collection_cache = set()

        self.default_collection = default_collection
        self._default_vector_size = vector_size
        self._default_distance_metric = distance_metric

        logger.info(
            f"Initialized QdrantVectorStore with config: path={path}, host={host}, url={url}"
        )

    def _distance_metric_to_qdrant(self, metric: DistanceMetric) -> Distance:
        """Convert framework distance metric to Qdrant distance."""
        mapping = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT,
            DistanceMetric.MANHATTAN: Distance.MANHATTAN,
        }
        return mapping.get(metric, Distance.COSINE)

    def _point_to_search_result(self, point) -> MemorySearchResult:
        """Convert Qdrant point to MemorySearchResult."""
        payload = getattr(point, "payload", {}) or {}
        # Allow content to be stored in common keys or fallback to empty string
        content = payload.get("content") or payload.get("text") or payload.get("data") or ""
        memory_type = payload.get("memory_type", "episodic")
        created_at = payload.get("created_at")
        updated_at = payload.get("updated_at")
        return MemorySearchResult(
            id=str(point.id),
            content=content,
            score=float(getattr(point, "score", 1.0) or 0.0),
            memory_type=memory_type,
            metadata=payload,
            vector=getattr(point, "vector", None),
            user_id=payload.get("user_id"),
            agent_id=payload.get("agent_id"),
            created_at=datetime.fromisoformat(created_at) if isinstance(created_at, str) else None,
            updated_at=datetime.fromisoformat(updated_at) if isinstance(updated_at, str) else None,
        )

    def _build_qdrant_filter(self, filters: dict[str, Any] | None) -> Filter | None:
        """Build Qdrant filter from dictionary."""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            # Qdrant filtering supports str/int/bool for exact match (avoid float ambig)
            if isinstance(value, (str, int, bool)):
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=conditions) if conditions else None

    # --- BaseStore abstract method implementations ---

    async def add(
        self,
        content: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str = "episodic",
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Add a new memory to Qdrant."""
        if not content:
            raise ValueError("content cannot be empty")
        
        collection = collection_name or self.default_collection
        
        # Ensure collection exists
        if not await self.collection_exists(collection):
            if self._default_vector_size is None:
                raise ValueError("Vector size must be specified at init to auto-create collection")
            await self.create_collection(
                collection, self._default_vector_size, self._default_distance_metric
            )

        # Generate embedding
        embedding = await self._agenerate_embedding(content)
        point_id = str(uuid.uuid4())
        payload = {
            "content": content,
            "user_id": user_id,
            "agent_id": agent_id,
            "memory_type": memory_type,
            "category": category,
            "created_at": datetime.now().isoformat(),
            **(metadata or {}),
        }
        
        await self._insert_vector(collection, embedding, payload, point_id)
        return point_id

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
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        """Search memories by content similarity."""
        collection = collection_name or self.default_collection
        
        # Build filters combining user/agent/memory_type/category
        combined_filters: dict[str, Any] = {}
        if user_id:
            combined_filters["user_id"] = user_id
        if agent_id:
            combined_filters["agent_id"] = agent_id
        if memory_type:
            combined_filters["memory_type"] = memory_type
        if category:
            combined_filters["category"] = category
        if filters:
            combined_filters.update(filters)
        
        # Generate query embedding
        embedding = await self._agenerate_embedding(query or "")
        
        # Perform vector search
        return await self._vector_search(
            collection,
            embedding,
            limit=limit,
            filters=combined_filters if combined_filters else None,
            score_threshold=score_threshold,
        )

    async def get(
        self, memory_id: str, collection_name: str | None = None, **_
    ) -> MemorySearchResult | None:
        """Get a specific memory by ID."""
        collection = collection_name or self.default_collection
        return await self._get_vector(collection, memory_id)

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Update an existing memory."""
        collection = collection_name or self.default_collection
        existing = await self._get_vector(collection, memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")
        
        payload = existing.metadata.copy() if existing.metadata else {}
        vector = None
        
        if content is not None:
            payload["content"] = content
            payload["updated_at"] = datetime.now().isoformat()
            vector = await self._agenerate_embedding(content)
        
        if metadata:
            payload.update(metadata)
        
        await self._update_vector(collection, memory_id, vector=vector, payload=payload)

    async def delete(self, memory_id: str, collection_name: str | None = None, **_) -> None:
        """Delete a memory by ID."""
        collection = collection_name or self.default_collection
        await self._delete_vector(collection, memory_id)

    # --- Raw vector operations (simplified, async-only) ---

    async def create_collection(
        self,
        name: str,
        vector_size: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any,
    ) -> None:
        """Create a new vector collection."""
        try:
            await self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size, distance=self._distance_metric_to_qdrant(distance_metric)
                ),
            )
            self._collection_cache.add(name)
            logger.info(f"Created collection '{name}' with {vector_size}D vectors")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.warning(f"Collection '{name}' already exists")
                self._collection_cache.add(name)
            else:
                logger.error(f"Failed to create collection '{name}': {e}")
                raise

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        if name in self._collection_cache:
            return True

        try:
            collections = await self.client.get_collections()
            exists = name in [col.name for col in collections.collections]
            if exists:
                self._collection_cache.add(name)
            return exists
        except Exception as e:
            logger.error(f"Failed to check collection existence '{name}': {e}")
            return False

    async def list_collections(self) -> list[str]:
        """List all collections."""
        try:
            collections = await self.client.get_collections()
            names = [col.name for col in collections.collections]
            self._collection_cache.update(names)
            return names
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    async def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            await self.client.delete_collection(collection_name=name)
            self._collection_cache.discard(name)
            logger.info(f"Deleted collection '{name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            raise

    async def _insert_vector(
        self,
        collection_name: str,
        vector: list[float],
        payload: dict[str, Any] | None = None,
        point_id: str | None = None,
    ) -> str:
        """Insert a single vector into collection."""
        point_id = point_id or str(uuid.uuid4())
        point = PointStruct(id=point_id, vector=vector, payload=payload or {})

        try:
            await self.client.upsert(collection_name=collection_name, points=[point])
            logger.debug(f"Inserted vector {point_id} into '{collection_name}'")
            return point_id
        except Exception as e:
            logger.error(f"Failed to insert vector into '{collection_name}': {e}")
            raise

    async def _vector_search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[MemorySearchResult]:
        """Search for similar vectors."""
        try:
            qdrant_filter = self._build_qdrant_filter(filters)

            results = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )

            return [self._point_to_search_result(point) for point in results]
        except Exception as e:
            logger.error(f"Search failed in '{collection_name}': {e}")
            raise

    async def _get_vector(self, collection_name: str, vector_id: str) -> MemorySearchResult | None:
        """Get vector by ID."""
        try:
            result = await self.client.retrieve(
                collection_name=collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=True,
            )

            if result:
                return self._point_to_search_result(result[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get vector '{vector_id}' from '{collection_name}': {e}")
            return None

    async def _update_vector(
        self,
        collection_name: str,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Update vector and/or payload."""
        try:
            if vector is not None:
                # Update vector and payload
                await self.client.upsert(
                    collection_name=collection_name,
                    points=[PointStruct(id=vector_id, vector=vector, payload=payload or {})],
                )
            elif payload is not None:
                # Update only payload
                await self.client.set_payload(
                    collection_name=collection_name, payload=payload, points=[vector_id]
                )

            logger.debug(f"Updated vector '{vector_id}' in '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to update vector '{vector_id}' in '{collection_name}': {e}")
            raise

    async def _delete_vector(self, collection_name: str, vector_id: str) -> None:
        """Delete vector by ID."""
        try:
            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[vector_id]),
            )
            logger.debug(f"Deleted vector '{vector_id}' from '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete vector '{vector_id}' from '{collection_name}': {e}")
            raise

    # --- Statistics and Management ---

    async def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Get collection statistics."""
        try:
            info = await self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "exists": True,
                "vectors_count": info.vectors_count or 0,
                "points_count": info.points_count or 0,
                "indexed_vectors_count": info.indexed_vectors_count or 0,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
            }
        except Exception as e:
            logger.error(f"Failed to get stats for '{collection_name}': {e}")
            return {"name": collection_name, "exists": False, "error": str(e)}

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self.client, "close"):
                await self.client.close()
            logger.info("Qdrant vector store cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory functions for convenience


def create_local_qdrant_vector_store(
    path: str = "./qdrant_data", **kwargs
) -> QdrantVectorStore:
    """Create a local file-based Qdrant vector store."""
    return QdrantVectorStore(path=path, **kwargs)


def create_remote_qdrant_vector_store(
    host: str = "localhost", port: int = 6333, api_key: str | None = None, **kwargs
) -> QdrantVectorStore:
    """Create a remote Qdrant vector store."""
    return QdrantVectorStore(host=host, port=port, api_key=api_key, **kwargs)


def create_cloud_qdrant_vector_store(url: str, api_key: str, **kwargs) -> QdrantVectorStore:
    """Create a cloud Qdrant vector store."""
    return QdrantVectorStore(url=url, api_key=api_key, **kwargs)
