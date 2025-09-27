"""
Qdrant Vector Store Implementation for PyAgenity Framework

This module provides a concrete implementation of BaseStore using Qdrant
as the backend vector database. Supports both local and cloud Qdrant deployments.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, cast

from .base_store import BaseStore, DistanceMetric, MemorySearchResult


try:
    from qdrant_client import AsyncQdrantClient, QdrantClient
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
    Qdrant-based vector store implementation.

    Features:
    - Support for both sync and async operations
    - Local and cloud Qdrant deployment support
    - Efficient vector similarity search
    - Collection management with automatic creation
    - Rich metadata filtering capabilities
    - Message-specific convenience methods

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
            **kwargs: Additional client parameters
        """
        # Initialize sync client
        if path:
            self.client = QdrantClient(path=path, **kwargs)
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key, **kwargs)
        else:
            host = host or "localhost"
            port = port or 6333
            self.client = QdrantClient(host=host, port=port, api_key=api_key, **kwargs)

        # Initialize async client with same parameters
        if path:
            self.async_client = AsyncQdrantClient(path=path, **kwargs)
        elif url:
            self.async_client = AsyncQdrantClient(url=url, api_key=api_key, **kwargs)
        else:
            self.async_client = AsyncQdrantClient(host=host, port=port, api_key=api_key, **kwargs)

        # Cache for collection existence checks
        self._collection_cache = set()

        self.default_collection = default_collection
        self._default_vector_size = vector_size
        self._default_distance_metric = distance_metric

        # Auto-create default collection if vector_size provided
        if vector_size is not None:
            try:
                self.create_collection(default_collection, vector_size, distance_metric)
            except Exception as e:
                logger.debug(f"Default collection creation skipped/failed: {e}")

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

    def add(
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
        if not content:
            raise ValueError("content cannot be empty")
        collection = collection_name or self.default_collection
        if not self.collection_exists(collection):
            if self._default_vector_size is None:
                raise ValueError("Vector size must be specified at init to auto-create collection")
            self.create_collection(
                collection, self._default_vector_size, self._default_distance_metric
            )

        # Generate embedding
        embedding = self._generate_embedding(content)
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
        self.insert(collection, embedding, payloads=payload, ids=point_id)
        return point_id

    async def aadd(
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
        collection = collection_name or self.default_collection
        if not await self.acollection_exists(collection):
            if self._default_vector_size is None:
                raise ValueError("Vector size must be specified at init to auto-create collection")
            await self.acreate_collection(
                collection, self._default_vector_size, self._default_distance_metric
            )
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
        await self.ainsert(collection, embedding, payloads=payload, ids=point_id)
        return point_id

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
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
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
        embedding = self._generate_embedding(query or "")
        results = self.vector_search(
            collection,
            embedding,
            limit=limit,
            filters=combined_filters if combined_filters else None,
            score_threshold=score_threshold,
        )
        return results

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
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        collection = collection_name or self.default_collection
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
        embedding = await self._agenerate_embedding(query or "")
        return await self.avector_search(
            collection,
            embedding,
            limit=limit,
            filters=combined_filters if combined_filters else None,
            score_threshold=score_threshold,
        )

    def get(
        self, memory_id: str, collection_name: str | None = None, **_
    ) -> MemorySearchResult | None:
        collection = collection_name or self.default_collection
        return self.vector_get(collection, memory_id)

    async def aget(
        self, memory_id: str, collection_name: str | None = None, **_
    ) -> MemorySearchResult | None:
        collection = collection_name or self.default_collection
        return await self.avector_get(collection, memory_id)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        collection = collection_name or self.default_collection
        existing = self.vector_get(collection, memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")
        payload = existing.metadata.copy() if existing.metadata else {}
        if content is not None:
            payload["content"] = content
            payload["updated_at"] = datetime.now().isoformat()
            vector = self._generate_embedding(content)
        else:
            vector = None
        if metadata:
            payload.update(metadata)
        self.vector_update(collection, memory_id, vector=vector, payload=payload)

    async def aupdate(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        collection = collection_name or self.default_collection
        existing = await self.avector_get(collection, memory_id)
        if not existing:
            raise ValueError(f"Memory {memory_id} not found")
        payload = existing.metadata.copy() if existing.metadata else {}
        if content is not None:
            payload["content"] = content
            payload["updated_at"] = datetime.now().isoformat()
            vector = await self._agenerate_embedding(content)
        else:
            vector = None
        if metadata:
            payload.update(metadata)
        await self.avector_update(collection, memory_id, vector=vector, payload=payload)

    def delete(self, memory_id: str, collection_name: str | None = None, **_) -> None:
        collection = collection_name or self.default_collection
        self.vector_delete(collection, memory_id)

    async def adelete(self, memory_id: str, collection_name: str | None = None, **_) -> None:
        collection = collection_name or self.default_collection
        await self.avector_delete(collection, memory_id)

    # --- Raw vector operations (original API preserved with vector_ prefix) ---

    # Collection Management

    def create_collection(
        self,
        name: str,
        vector_size: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any,
    ) -> None:
        """Create a new vector collection."""
        try:
            self.client.create_collection(
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

    async def acreate_collection(
        self,
        name: str,
        vector_size: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any,
    ) -> None:
        """Async create collection."""
        try:
            await self.async_client.create_collection(
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

    def list_collections(self) -> list[str]:
        """List all collections."""
        try:
            collections = self.client.get_collections()
            names = [col.name for col in collections.collections]
            self._collection_cache.update(names)
            return names
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    async def alist_collections(self) -> list[str]:
        """Async list collections."""
        try:
            collections = await self.async_client.get_collections()
            names = [col.name for col in collections.collections]
            self._collection_cache.update(names)
            return names
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name=name)
            self._collection_cache.discard(name)
            logger.info(f"Deleted collection '{name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            raise

    async def adelete_collection(self, name: str) -> None:
        """Async delete collection."""
        try:
            await self.async_client.delete_collection(collection_name=name)
            self._collection_cache.discard(name)
            logger.info(f"Deleted collection '{name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            raise

    def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        if name in self._collection_cache:
            return True

        try:
            collections = self.client.get_collections()
            exists = name in [col.name for col in collections.collections]
            if exists:
                self._collection_cache.add(name)
            return exists
        except Exception as e:
            logger.error(f"Failed to check collection existence '{name}': {e}")
            return False

    async def acollection_exists(self, name: str) -> bool:
        """Async check collection exists."""
        if name in self._collection_cache:
            return True

        try:
            collections = await self.async_client.get_collections()
            exists = name in [col.name for col in collections.collections]
            if exists:
                self._collection_cache.add(name)
            return exists
        except Exception as e:
            logger.error(f"Failed to check collection existence '{name}': {e}")
            return False

    # Vector Operations

    def insert(
        self,
        collection_name: str,
        vectors: list[float] | list[list[float]],
        payloads: dict[str, Any] | list[dict[str, Any]] | None = None,
        ids: str | list[str] | None = None,
    ) -> list[str]:
        """Insert vectors into collection (low-level)."""
        # Normalize vectors to list[list[float]]
        if vectors and isinstance(vectors[0], (int, float)):
            vectors = cast(list[list[float]], [vectors])  # type: ignore[list-item]
        vectors_list: list[list[float]] = cast(list[list[float]], vectors)

        if payloads is None:
            payloads = [{}] * len(vectors_list)
        elif isinstance(payloads, dict):
            payloads = [payloads]

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors_list))]
        elif isinstance(ids, str):
            ids = [ids]

        # Create points
        points = []
        for _i, (vector, payload, point_id) in enumerate(
            zip(vectors_list, payloads, ids, strict=False)
        ):
            points.append(PointStruct(id=point_id, vector=vector, payload=payload or {}))

        try:
            self.client.upsert(collection_name=collection_name, points=points)
            logger.debug(f"Inserted {len(points)} vectors into '{collection_name}'")
            return ids
        except Exception as e:
            logger.error(f"Failed to insert vectors into '{collection_name}': {e}")
            raise

    async def ainsert(
        self,
        collection_name: str,
        vectors: list[float] | list[list[float]],
        payloads: dict[str, Any] | list[dict[str, Any]] | None = None,
        ids: str | list[str] | None = None,
    ) -> list[str]:
        """Async insert vectors."""
        # Normalize inputs to lists
        if vectors and isinstance(vectors[0], (int, float)):
            vectors = cast(list[list[float]], [vectors])  # type: ignore[list-item]
        vectors_list: list[list[float]] = cast(list[list[float]], vectors)

        if payloads is None:
            payloads = [{}] * len(vectors_list)
        elif isinstance(payloads, dict):
            payloads = [payloads]

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors_list))]
        elif isinstance(ids, str):
            ids = [ids]

        # Create points
        points = []
        for _i, (vector, payload, point_id) in enumerate(
            zip(vectors_list, payloads, ids, strict=False)
        ):
            points.append(PointStruct(id=point_id, vector=vector, payload=payload or {}))

        try:
            await self.async_client.upsert(collection_name=collection_name, points=points)
            logger.debug(f"Inserted {len(points)} vectors into '{collection_name}'")
            return ids
        except Exception as e:
            logger.error(f"Failed to insert vectors into '{collection_name}': {e}")
            raise

    def vector_search(
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

            results = self.client.search(
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

    async def avector_search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[MemorySearchResult]:
        """Async search for similar vectors."""
        try:
            qdrant_filter = self._build_qdrant_filter(filters)

            results = await self.async_client.search(
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

    def vector_get(self, collection_name: str, vector_id: str) -> MemorySearchResult | None:
        """Get vector by ID."""
        try:
            result = self.client.retrieve(
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

    async def avector_get(self, collection_name: str, vector_id: str) -> MemorySearchResult | None:
        """Async get vector by ID."""
        try:
            result = await self.async_client.retrieve(
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

    def vector_update(
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
                self.client.upsert(
                    collection_name=collection_name,
                    points=[PointStruct(id=vector_id, vector=vector, payload=payload or {})],
                )
            elif payload is not None:
                # Update only payload
                self.client.set_payload(
                    collection_name=collection_name, payload=payload, points=[vector_id]
                )

            logger.debug(f"Updated vector '{vector_id}' in '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to update vector '{vector_id}' in '{collection_name}': {e}")
            raise

    async def avector_update(
        self,
        collection_name: str,
        vector_id: str,
        vector: list[float] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Async update vector."""
        try:
            if vector is not None:
                # Update vector and payload
                await self.async_client.upsert(
                    collection_name=collection_name,
                    points=[PointStruct(id=vector_id, vector=vector, payload=payload or {})],
                )
            elif payload is not None:
                # Update only payload
                await self.async_client.set_payload(
                    collection_name=collection_name, payload=payload, points=[vector_id]
                )

            logger.debug(f"Updated vector '{vector_id}' in '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to update vector '{vector_id}' in '{collection_name}': {e}")
            raise

    def vector_delete(self, collection_name: str, vector_id: str) -> None:
        """Delete vector by ID."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[vector_id]),
            )
            logger.debug(f"Deleted vector '{vector_id}' from '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete vector '{vector_id}' from '{collection_name}': {e}")
            raise

    async def avector_delete(self, collection_name: str, vector_id: str) -> None:
        """Async delete vector."""
        try:
            await self.async_client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[vector_id]),
            )
            logger.debug(f"Deleted vector '{vector_id}' from '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete vector '{vector_id}' from '{collection_name}': {e}")
            raise

    # Utility Methods

    def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "exists": True,
                "vectors_count": getattr(info, "vectors_count", 0) or 0,
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0) or 0,
                "status": getattr(info, "status", None),
                "optimizer_status": getattr(info, "optimizer_status", None),
                "disk_data_size": getattr(info, "disk_data_size", None),
                "ram_data_size": getattr(info, "ram_data_size", None),
            }
        except Exception as e:
            logger.error(f"Failed to get stats for '{collection_name}': {e}")
            return {"name": collection_name, "exists": False, "error": str(e)}

    async def aget_collection_stats(self, collection_name: str) -> dict[str, Any]:
        """Async get collection stats."""
        try:
            info = await self.async_client.get_collection(collection_name)
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

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self.client, "close"):
                self.client.close()
            logger.info("Qdrant vector store cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def acleanup(self) -> None:
        """Async cleanup."""
        try:
            if hasattr(self.async_client, "close"):
                await self.async_client.close()
            logger.info("Qdrant vector store cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory functions for convenience


def create_local_qdrant_vector_store(path: str = "./qdrant_data") -> QdrantVectorStore:
    """Create a local file-based Qdrant vector store."""
    return QdrantVectorStore(path=path)


def create_remote_qdrant_vector_store(
    host: str = "localhost", port: int = 6333, api_key: str | None = None
) -> QdrantVectorStore:
    """Create a remote Qdrant vector store."""
    return QdrantVectorStore(host=host, port=port, api_key=api_key)


def create_cloud_qdrant_vector_store(url: str, api_key: str) -> QdrantVectorStore:
    """Create a cloud Qdrant vector store."""
    return QdrantVectorStore(url=url, api_key=api_key)
