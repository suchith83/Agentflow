"""
Qdrant Vector Store Implementation for PyAgenity Framework

This module provides a concrete implementation of VectorStoreBase using Qdrant
as the backend vector database. Supports both local and cloud Qdrant deployments.
"""

import logging
import uuid
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .vector_base_store import (
    VectorStoreBase,
    DistanceMetric,
    VectorSearchResult,
    MemoryRecord
)

try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        CollectionInfo,
        UpdateStatus
    )
except ImportError:
    raise ImportError(
        "Qdrant client not installed. Install with: pip install qdrant-client"
    )

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStoreBase):
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
        store = QdrantVectorStore(
            url="https://xyz.qdrant.io",
            api_key="your-api-key"
        )
        ```
    """
    
    def __init__(
        self,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any
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
        
        logger.info(f"Initialized QdrantVectorStore with config: path={path}, host={host}, url={url}")
    
    def _distance_metric_to_qdrant(self, metric: DistanceMetric) -> Distance:
        """Convert framework distance metric to Qdrant distance."""
        mapping = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT,
            DistanceMetric.MANHATTAN: Distance.MANHATTAN
        }
        return mapping.get(metric, Distance.COSINE)
    
    def _point_to_search_result(self, point) -> VectorSearchResult:
        """Convert Qdrant point to VectorSearchResult."""
        return VectorSearchResult(
            id=str(point.id),
            score=getattr(point, 'score', 1.0),
            payload=point.payload or {},
            vector=getattr(point, 'vector', None)
        )
    
    def _build_qdrant_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Build Qdrant filter from dictionary."""
        if not filters:
            return None
        
        conditions = []
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions) if conditions else None
    
    # Collection Management
    
    def create_collection(
        self,
        name: str,
        vector_size: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any
    ) -> None:
        """Create a new vector collection."""
        try:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=self._distance_metric_to_qdrant(distance_metric)
                )
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
        **kwargs: Any
    ) -> None:
        """Async create collection."""
        try:
            await self.async_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=self._distance_metric_to_qdrant(distance_metric)
                )
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
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.get_collections()
            names = [col.name for col in collections.collections]
            self._collection_cache.update(names)
            return names
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
    
    async def alist_collections(self) -> List[str]:
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
        vectors: Union[List[float], List[List[float]]],
        payloads: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """Insert vectors into collection."""
        # Normalize inputs to lists
        if isinstance(vectors[0], (int, float)):
            vectors = [vectors]
        
        if payloads is None:
            payloads = [{}] * len(vectors)
        elif isinstance(payloads, dict):
            payloads = [payloads]
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif isinstance(ids, str):
            ids = [ids]
        
        # Create points
        points = []
        for i, (vector, payload, point_id) in enumerate(zip(vectors, payloads, ids)):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload or {}
                )
            )
        
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.debug(f"Inserted {len(points)} vectors into '{collection_name}'")
            return ids
        except Exception as e:
            logger.error(f"Failed to insert vectors into '{collection_name}': {e}")
            raise
    
    async def ainsert(
        self,
        collection_name: str,
        vectors: Union[List[float], List[List[float]]],
        payloads: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """Async insert vectors."""
        # Normalize inputs to lists
        if isinstance(vectors[0], (int, float)):
            vectors = [vectors]
        
        if payloads is None:
            payloads = [{}] * len(vectors)
        elif isinstance(payloads, dict):
            payloads = [payloads]
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif isinstance(ids, str):
            ids = [ids]
        
        # Create points
        points = []
        for i, (vector, payload, point_id) in enumerate(zip(vectors, payloads, ids)):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload or {}
                )
            )
        
        try:
            await self.async_client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.debug(f"Inserted {len(points)} vectors into '{collection_name}'")
            return ids
        except Exception as e:
            logger.error(f"Failed to insert vectors into '{collection_name}': {e}")
            raise
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
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
                with_vectors=False
            )
            
            return [self._point_to_search_result(point) for point in results]
        except Exception as e:
            logger.error(f"Search failed in '{collection_name}': {e}")
            raise
    
    async def asearch(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
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
                with_vectors=False
            )
            
            return [self._point_to_search_result(point) for point in results]
        except Exception as e:
            logger.error(f"Search failed in '{collection_name}': {e}")
            raise
    
    def get(self, collection_name: str, vector_id: str) -> Optional[VectorSearchResult]:
        """Get vector by ID."""
        try:
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=True
            )
            
            if result:
                return self._point_to_search_result(result[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get vector '{vector_id}' from '{collection_name}': {e}")
            return None
    
    async def aget(self, collection_name: str, vector_id: str) -> Optional[VectorSearchResult]:
        """Async get vector by ID."""
        try:
            result = await self.async_client.retrieve(
                collection_name=collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=True
            )
            
            if result:
                return self._point_to_search_result(result[0])
            return None
        except Exception as e:
            logger.error(f"Failed to get vector '{vector_id}' from '{collection_name}': {e}")
            return None
    
    def update(
        self,
        collection_name: str,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update vector and/or payload."""
        try:
            if vector is not None:
                # Update vector and payload
                self.client.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=vector_id,
                            vector=vector,
                            payload=payload or {}
                        )
                    ]
                )
            elif payload is not None:
                # Update only payload
                self.client.set_payload(
                    collection_name=collection_name,
                    payload=payload,
                    points=[vector_id]
                )
            
            logger.debug(f"Updated vector '{vector_id}' in '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to update vector '{vector_id}' in '{collection_name}': {e}")
            raise
    
    async def aupdate(
        self,
        collection_name: str,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Async update vector."""
        try:
            if vector is not None:
                # Update vector and payload
                await self.async_client.upsert(
                    collection_name=collection_name,
                    points=[
                        PointStruct(
                            id=vector_id,
                            vector=vector,
                            payload=payload or {}
                        )
                    ]
                )
            elif payload is not None:
                # Update only payload
                await self.async_client.set_payload(
                    collection_name=collection_name,
                    payload=payload,
                    points=[vector_id]
                )
            
            logger.debug(f"Updated vector '{vector_id}' in '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to update vector '{vector_id}' in '{collection_name}': {e}")
            raise
    
    def delete(self, collection_name: str, vector_id: str) -> None:
        """Delete vector by ID."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[vector_id]
                )
            )
            logger.debug(f"Deleted vector '{vector_id}' from '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete vector '{vector_id}' from '{collection_name}': {e}")
            raise
    
    async def adelete(self, collection_name: str, vector_id: str) -> None:
        """Async delete vector."""
        try:
            await self.async_client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[vector_id]
                )
            )
            logger.debug(f"Deleted vector '{vector_id}' from '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete vector '{vector_id}' from '{collection_name}': {e}")
            raise
    
    # Utility Methods
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "exists": True,
                "vectors_count": info.vectors_count or 0,
                "indexed_vectors_count": info.indexed_vectors_count or 0,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "disk_data_size": info.disk_data_size,
                "ram_data_size": info.ram_data_size
            }
        except Exception as e:
            logger.error(f"Failed to get stats for '{collection_name}': {e}")
            return {
                "name": collection_name,
                "exists": False,
                "error": str(e)
            }
    
    async def aget_collection_stats(self, collection_name: str) -> Dict[str, Any]:
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
            return {
                "name": collection_name,
                "exists": False,
                "error": str(e)
            }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            logger.info("Qdrant vector store cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def acleanup(self) -> None:
        """Async cleanup."""
        try:
            if hasattr(self.async_client, 'close'):
                await self.async_client.close()
            logger.info("Qdrant vector store cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Factory functions for convenience

def create_local_qdrant_vector_store(path: str = "./qdrant_data") -> QdrantVectorStore:
    """Create a local file-based Qdrant vector store."""
    return QdrantVectorStore(path=path)


def create_remote_qdrant_vector_store(
    host: str = "localhost",
    port: int = 6333,
    api_key: Optional[str] = None
) -> QdrantVectorStore:
    """Create a remote Qdrant vector store."""
    return QdrantVectorStore(host=host, port=port, api_key=api_key)


def create_cloud_qdrant_vector_store(url: str, api_key: str) -> QdrantVectorStore:
    """Create a cloud Qdrant vector store."""
    return QdrantVectorStore(url=url, api_key=api_key)