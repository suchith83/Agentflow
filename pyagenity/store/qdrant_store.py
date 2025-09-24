import logging
import asyncio
import uuid
from typing import Any, TypeVar, Optional
from dataclasses import dataclass, asdict
import json
import hashlib
from datetime import datetime
from .base_store import BaseStore

try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
except ImportError:
    raise ImportError("Please install qdrant-client: pip install qdrant-client")

# Generic type variable for extensible data types
DataT = TypeVar("DataT")
logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Standardized memory item structure."""
    id: str
    content: str
    metadata: dict[str, Any]
    timestamp: str
    embedding: Optional[list[float]] = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'MemoryItem':
        return cls(**data)

class QdrantStore(BaseStore[DataT]):
    """Qdrant implementation for long-term memory storage.
    
    Async-first implementation where sync methods wrap async ones.
    
    Features:
    - Vector similarity search for semantic retrieval
    - Metadata filtering for structured queries
    - Configurable collections per conversation/user
    - Async-first design with sync wrappers
    - Automatic embedding generation
    """
    
    def __init__(
        self,
        async_client: AsyncQdrantClient,
        embedding_function: callable,
        collection_prefix: str = "memory",
        vector_size: int = 768,  # Default for Gemini embeddings
        distance_metric: Distance = Distance.COSINE,
        max_retries: int = 3,
        default_limit: int = 10
    ):
        """Initialize Qdrant store with async client.
        
        Args:
            async_client: AsyncQdrantClient instance
            embedding_function: Function to generate embeddings from text
            collection_prefix: Prefix for collection names
            vector_size: Dimension of embedding vectors
            distance_metric: Distance metric for similarity search
            max_retries: Number of retry attempts for operations
            default_limit: Default limit for search results
        """
        self.async_client = async_client
        self.embedding_function = embedding_function
        self.collection_prefix = collection_prefix
        self.vector_size = vector_size
        self.distance_metric = distance_metric
        self.max_retries = max_retries
        self.default_limit = default_limit
        self._collections_cache = set()
        
        # Create event loop for sync operations if none exists
        self._loop = None
    
    def _get_or_create_event_loop(self):
        """Get or create event loop for sync operations."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    def _run_async(self, coro):
        """Helper to run async function from sync context."""
        try:
            # Try to get the current loop
            loop = asyncio.get_running_loop()
            # If we're already in an async context, we can't use run_until_complete
            # This should not happen in normal usage, but let's handle it gracefully
            raise RuntimeError("Cannot run async function from within async context. Use await instead.")
        except RuntimeError:
            # No running loop, safe to create one
            return asyncio.run(coro)
    
    def _get_collection_name(self, config: dict[str, Any]) -> str:
        """Generate collection name from config."""
        # Use conversation_id, user_id, or session_id from config
        identifiers = []
        for key in ['conversation_id', 'user_id', 'session_id', 'thread_id']:
            if key in config:
                identifiers.append(str(config[key]))
        
        if not identifiers:
            identifiers.append('default')
        
        # Create a hash for very long identifiers to avoid collection name limits
        identifier = "_".join(identifiers)
        if len(identifier) > 50:  # Qdrant collection name limit consideration
            identifier = hashlib.md5(identifier.encode()).hexdigest()
        
        return f"{self.collection_prefix}_{identifier}"
    
    async def _aensure_collection_exists(self, collection_name: str) -> None:
        """Ensure collection exists, create if not."""
        if collection_name in self._collections_cache:
            return
        
        try:
            # Check if collection exists
            collections = await self.async_client.get_collections()
            existing_names = [c.name for c in collections.collections]
            
            if collection_name not in existing_names:
                # Create collection
                await self.async_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric
                    )
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            
            self._collections_cache.add(collection_name)
            
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise
    
    def _extract_text_content(self, info: DataT) -> str:
        """Extract text content from data for embedding."""
        if isinstance(info, str):
            return info
        elif isinstance(info, dict):
            # Priority order for text extraction
            for field in ['content', 'text', 'message', 'summary', 'description']:
                if field in info and isinstance(info[field], str):
                    return info[field]
            
            # Fallback: concatenate string values
            text_parts = []
            for key, value in info.items():
                if isinstance(value, str) and key not in ['id', 'timestamp']:
                    text_parts.append(f"{key}: {value}")
            
            return " | ".join(text_parts)
        elif hasattr(info, '__dict__'):
            return self._extract_text_content(info.__dict__)
        else:
            return str(info)
    
    async def _acreate_memory_item(self, config: dict[str, Any], info: DataT) -> MemoryItem:
        """Create standardized memory item."""
        content = self._extract_text_content(info)
        
        # Generate embedding
        embedding = None
        if content:
            # Handle both sync and async embedding functions
            if asyncio.iscoroutinefunction(self.embedding_function):
                embedding_result = await self.embedding_function([content])
            else:
                embedding_result = self.embedding_function([content])
            embedding = embedding_result[0] if embedding_result else None
        
        # Extract metadata
        metadata = {}
        if isinstance(info, dict):
            metadata = {k: v for k, v in info.items() if k not in ['content', 'text']}
        
        # Add config metadata
        metadata.update({
            'config': config,
            'collection': self._get_collection_name(config)
        })
        
        return MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat(),
            embedding=embedding
        )
    
    async def _aretry_operation(self, operation: callable, *args, **kwargs) -> Any:
        """Retry async operation with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                else:
                    return operation(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (2 ** attempt) * 0.1
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                await asyncio.sleep(wait_time)

    # ASYNC METHODS (Primary implementations)
    async def aupdate_memory(self, config: dict[str, Any], info: DataT) -> None:
        """Store a single piece of information."""
        collection_name = self._get_collection_name(config)
        await self._aensure_collection_exists(collection_name)
        
        memory_item = await self._acreate_memory_item(config, info)
        
        if memory_item.embedding is None:
            raise ValueError("Failed to generate embedding for memory item")
        
        point = PointStruct(
            id=memory_item.id,
            vector=memory_item.embedding,
            payload=memory_item.metadata
        )
        
        async def _aupsert():
            return await self.async_client.upsert(
                collection_name=collection_name,
                points=[point]
            )
        
        await self._aretry_operation(_aupsert)
        logger.debug(f"Stored memory item {memory_item.id} in {collection_name}")
    
    async def aget_memory(self, config: dict[str, Any]) -> DataT | None:
        """Retrieve the most recent piece of information."""
        collection_name = self._get_collection_name(config)
        
        try:
            # Search for most recent item by timestamp
            results = await self.async_client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if not results[0]:  # Empty results
                return None
            
            # Get the most recent item
            point = max(results[0], key=lambda p: p.payload.get('timestamp', ''))
            return point.payload
            
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            return None
    
    async def adelete_memory(self, config: dict[str, Any]) -> None:
        """Delete all information for the given config."""
        collection_name = self._get_collection_name(config)
        
        try:
            # Delete the entire collection
            await self.async_client.delete_collection(collection_name)
            self._collections_cache.discard(collection_name)
            logger.info(f"Deleted collection {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            raise
    
    async def arelated_memory(self, config: dict[str, Any], query: str) -> list[DataT]:
        """Retrieve related information using semantic search."""
        collection_name = self._get_collection_name(config)
        
        try:
            # Generate query embedding
            if asyncio.iscoroutinefunction(self.embedding_function):
                query_embedding_result = await self.embedding_function([query])
            else:
                query_embedding_result = self.embedding_function([query])
            query_embedding = query_embedding_result[0]
            
            # Perform similarity search
            results = await self.async_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=self.default_limit,
                with_payload=True
            )
            
            return [hit.payload for hit in results]
            
        except Exception as e:
            logger.error(f"Error searching related memory: {e}")
            return []
    
    async def arelease(self) -> None:
        """Clean up resources."""
        try:
            await self.async_client.close()
            logger.info("Qdrant async client connection closed")
        except Exception as e:
            logger.error(f"Error closing Qdrant async client: {e}")

    # SYNC METHODS (Wrappers around async methods)
    def update_memory(self, config: dict[str, Any], info: DataT) -> None:
        """Store a single piece of information (sync wrapper)."""
        return self._run_async(self.aupdate_memory(config, info))
    
    def get_memory(self, config: dict[str, Any]) -> DataT | None:
        """Retrieve a single piece of information (sync wrapper)."""
        return self._run_async(self.aget_memory(config))
    
    def delete_memory(self, config: dict[str, Any]) -> None:
        """Delete a single piece of information (sync wrapper)."""
        return self._run_async(self.adelete_memory(config))
    
    def related_memory(self, config: dict[str, Any], query: str) -> list[DataT]:
        """Retrieve related information (sync wrapper)."""
        return self._run_async(self.arelated_memory(config, query))
    
    def release(self) -> None:
        """Clean up resources (sync wrapper)."""
        return self._run_async(self.arelease())


# Updated Factory with async-first approach
class QdrantStoreFactory:
    """Factory for creating configured QdrantStore instances."""
    
    @staticmethod
    async def acreate_local_store(
        embedding_function: callable,
        path: str = "./qdrant_data",
        **kwargs
    ) -> QdrantStore:
        """Create a local file-based Qdrant store (async)."""
        async_client = AsyncQdrantClient(path=path)
        return QdrantStore(async_client=async_client, embedding_function=embedding_function, **kwargs)
    
    @staticmethod
    async def acreate_remote_store(
        embedding_function: callable,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        **kwargs
    ) -> QdrantStore:
        """Create a remote Qdrant store (async)."""
        async_client = AsyncQdrantClient(host=host, port=port, api_key=api_key)
        return QdrantStore(async_client=async_client, embedding_function=embedding_function, **kwargs)
    
    @staticmethod
    async def acreate_cloud_store(
        embedding_function: callable,
        url: str,
        api_key: str,
        **kwargs
    ) -> QdrantStore:
        """Create a Qdrant cloud store (async)."""
        async_client = AsyncQdrantClient(url=url, api_key=api_key)
        return QdrantStore(async_client=async_client, embedding_function=embedding_function, **kwargs)
    
    # Sync wrappers for the factory methods
    @staticmethod
    def create_local_store(
        embedding_function: callable,
        path: str = "./qdrant_data",
        **kwargs
    ) -> QdrantStore:
        """Create a local file-based Qdrant store (sync wrapper)."""
        return asyncio.run(QdrantStoreFactory.acreate_local_store(embedding_function, path, **kwargs))
    
    @staticmethod
    def create_remote_store(
        embedding_function: callable,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        **kwargs
    ) -> QdrantStore:
        """Create a remote Qdrant store (sync wrapper)."""
        return asyncio.run(QdrantStoreFactory.acreate_remote_store(embedding_function, host, port, api_key, **kwargs))
    
    @staticmethod
    def create_cloud_store(
        embedding_function: callable,
        url: str,
        api_key: str,
        **kwargs
    ) -> QdrantStore:
        """Create a Qdrant cloud store (sync wrapper)."""
        return asyncio.run(QdrantStoreFactory.acreate_cloud_store(embedding_function, url, api_key, **kwargs))


# Example usage showing both async and sync approaches
async def example_async_usage():
    """Example of async-first usage."""
    # Setup
    def embedding_fn(texts):
        return [[0.1, 0.2] * 768 for _ in texts]  # Mock embedding
    
    # Create store (async)
    store = await QdrantStoreFactory.acreate_local_store(embedding_fn)
    
    config = {"conversation_id": "conv_123"}
    data = {"content": "User prefers morning meetings", "type": "preference"}
    
    # Use async methods directly
    await store.aupdate_memory(config, data)
    memory = await store.aget_memory(config)
    related = await store.arelated_memory(config, "meeting preferences")
    
    print(f"Retrieved: {memory}")
    print(f"Related: {related}")
    
    await store.arelease()

def example_sync_usage():
    """Example of sync usage (wraps async internally)."""
    # Setup
    def embedding_fn(texts):
        return [[0.1, 0.2] * 768 for _ in texts]  # Mock embedding
    
    # Create store (sync - calls async internally)
    store = QdrantStoreFactory.create_local_store(embedding_fn)
    
    config = {"conversation_id": "conv_456"}
    data = {"content": "User likes afternoon calls", "type": "preference"}
    
    # Use sync methods (they wrap async internally)
    store.update_memory(config, data)
    memory = store.get_memory(config)
    related = store.related_memory(config, "call preferences")
    
    print(f"Retrieved: {memory}")
    print(f"Related: {related}")
    
    store.release()

if __name__ == "__main__":
    # Run async example
    print("Running async example:")
    asyncio.run(example_async_usage())
    
    print("\nRunning sync example:")
    example_sync_usage()