"""
Vector Store Base Class for PyAgenity Framework

This module provides the abstract base class for vector store implementations,
designed for long-term memory storage with semantic similarity search capabilities.
Inspired by mem0's vector store interface but adapted for PyAgenity's Message system.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pyagenity.utils.message import Message

# Generic type variable for extensible data types
DataT = TypeVar("DataT")

logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Supported distance metrics for vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search operation."""
    id: str
    score: float
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "score": self.score,
            "payload": self.payload,
            "vector": self.vector
        }


@dataclass
class MemoryRecord:
    """Standardized record for storing messages and data as long-term memory."""
    id: str
    content: str
    vector: Optional[List[float]]
    metadata: Dict[str, Any]
    timestamp: datetime
    source_type: str  # 'message', 'custom', etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "vector": self.vector,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "source_type": self.source_type
        }
    
    @classmethod
    def from_message(
        cls, 
        message: Message, 
        vector: Optional[List[float]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> "MemoryRecord":
        """Create a MemoryRecord from a PyAgenity Message."""
        metadata = {
            "role": message.role,
            "message_id": str(message.message_id),
            "timestamp": message.timestamp.isoformat() if message.timestamp else None,
            "has_tool_calls": bool(message.tools_calls),
            "has_reasoning": bool(message.reasoning),
            "token_usage": message.usages.model_dump() if message.usages else None,
            **(additional_metadata or {})
        }
        
        return cls(
            id=str(message.message_id),
            content=message.text(),
            vector=vector,
            metadata=metadata,
            timestamp=message.timestamp or datetime.now(),
            source_type="message"
        )


class VectorStoreBase(ABC, Generic[DataT]):
    """
    Abstract base class for vector store implementations.
    
    This class defines the interface for vector stores that can be used as long-term
    memory in PyAgenity agents. It supports both raw vector operations and convenience
    methods for storing PyAgenity Messages.
    
    Key Features:
    - Vector similarity search for semantic retrieval
    - Collection management for organizing data
    - Message-specific convenience methods
    - Async support throughout
    - Extensible filtering and metadata support
    """
    
    # --- Collection Management ---
    
    @abstractmethod
    def create_collection(
        self, 
        name: str, 
        vector_size: int, 
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any
    ) -> None:
        """
        Create a new vector collection.
        
        Args:
            name: Collection name
            vector_size: Dimension of vectors in this collection
            distance_metric: Distance metric for similarity calculations
            **kwargs: Implementation-specific parameters
            
        Raises:
            ValueError: If collection already exists or invalid parameters
            RuntimeError: If collection creation fails
        """
        pass
    
    @abstractmethod
    async def acreate_collection(
        self, 
        name: str, 
        vector_size: int, 
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any
    ) -> None:
        """Async version of create_collection."""
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        pass
    
    @abstractmethod
    async def alist_collections(self) -> List[str]:
        """Async version of list_collections."""
        pass
    
    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """
        Delete a collection and all its data.
        
        Args:
            name: Collection name to delete
            
        Raises:
            ValueError: If collection doesn't exist
        """
        pass
    
    @abstractmethod
    async def adelete_collection(self, name: str) -> None:
        """Async version of delete_collection."""
        pass
    
    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            name: Collection name to check
            
        Returns:
            True if collection exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def acollection_exists(self, name: str) -> bool:
        """Async version of collection_exists."""
        pass
    
    # --- Vector Operations ---
    
    @abstractmethod
    def insert(
        self,
        collection_name: str,
        vectors: Union[List[float], List[List[float]]],
        payloads: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """
        Insert vectors into a collection.
        
        Args:
            collection_name: Target collection
            vectors: Vector(s) to insert
            payloads: Optional metadata for each vector
            ids: Optional IDs for vectors (auto-generated if not provided)
            
        Returns:
            List of assigned IDs
            
        Raises:
            ValueError: If collection doesn't exist or dimension mismatch
        """
        pass
    
    @abstractmethod
    async def ainsert(
        self,
        collection_name: str,
        vectors: Union[List[float], List[List[float]]],
        payloads: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None
    ) -> List[str]:
        """Async version of insert."""
        pass
    
    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Collection to search in
            query_vector: Vector to find similarities for
            limit: Maximum number of results
            filters: Optional metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results ordered by similarity
            
        Raises:
            ValueError: If collection doesn't exist or dimension mismatch
        """
        pass
    
    @abstractmethod
    async def asearch(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[VectorSearchResult]:
        """Async version of search."""
        pass
    
    @abstractmethod
    def get(self, collection_name: str, vector_id: str) -> Optional[VectorSearchResult]:
        """
        Retrieve a vector by ID.
        
        Args:
            collection_name: Collection containing the vector
            vector_id: ID of the vector to retrieve
            
        Returns:
            Vector data if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def aget(self, collection_name: str, vector_id: str) -> Optional[VectorSearchResult]:
        """Async version of get."""
        pass
    
    @abstractmethod
    def update(
        self,
        collection_name: str,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update a vector and/or its payload.
        
        Args:
            collection_name: Collection containing the vector
            vector_id: ID of vector to update
            vector: New vector values (optional)
            payload: New payload data (optional)
            
        Raises:
            ValueError: If vector doesn't exist or dimension mismatch
        """
        pass
    
    @abstractmethod
    async def aupdate(
        self,
        collection_name: str,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Async version of update."""
        pass
    
    @abstractmethod
    def delete(self, collection_name: str, vector_id: str) -> None:
        """
        Delete a vector by ID.
        
        Args:
            collection_name: Collection containing the vector
            vector_id: ID of vector to delete
            
        Raises:
            ValueError: If vector doesn't exist
        """
        pass
    
    @abstractmethod
    async def adelete(self, collection_name: str, vector_id: str) -> None:
        """Async version of delete."""
        pass
    
    # --- Memory-Specific Convenience Methods ---
    
    def store_message(
        self,
        collection_name: str,
        message: Message,
        embedding_function: callable,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a PyAgenity Message as long-term memory.
        
        Args:
            collection_name: Target collection
            message: Message to store
            embedding_function: Function to generate embeddings from text
            additional_metadata: Extra metadata to attach
            
        Returns:
            ID of the stored memory record
        """
        # Extract text content and generate embedding
        content = message.text()
        if not content.strip():
            logger.warning("Empty message content, skipping storage")
            return ""
        
        # Generate embedding (handle both sync and async functions)
        if hasattr(embedding_function, '__call__'):
            embedding = embedding_function([content])[0]
        else:
            raise ValueError("embedding_function must be callable")
        
        # Create memory record
        record = MemoryRecord.from_message(message, embedding, additional_metadata)
        
        # Store in vector database
        return self.insert(
            collection_name=collection_name,
            vectors=embedding,
            payloads=record.metadata,
            ids=record.id
        )[0]
    
    async def astore_message(
        self,
        collection_name: str,
        message: Message,
        embedding_function: callable,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Async version of store_message."""
        # Extract text content and generate embedding
        content = message.text()
        if not content.strip():
            logger.warning("Empty message content, skipping storage")
            return ""
        
        # Generate embedding (handle both sync and async functions)
        import asyncio
        if asyncio.iscoroutinefunction(embedding_function):
            embedding_result = await embedding_function([content])
            embedding = embedding_result[0]
        else:
            embedding = embedding_function([content])[0]
        
        # Create memory record
        record = MemoryRecord.from_message(message, embedding, additional_metadata)
        
        # Store in vector database
        result = await self.ainsert(
            collection_name=collection_name,
            vectors=embedding,
            payloads=record.metadata,
            ids=record.id
        )
        return result[0]
    
    def recall_similar_messages(
        self,
        collection_name: str,
        query: str,
        embedding_function: callable,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        role_filter: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """
        Retrieve messages similar to the query.
        
        Args:
            collection_name: Collection to search in
            query: Text query to find similar messages for
            embedding_function: Function to generate query embedding
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            role_filter: Optional filter by message role
            
        Returns:
            List of similar message records
        """
        # Generate query embedding
        if hasattr(embedding_function, '__call__'):
            query_embedding = embedding_function([query])[0]
        else:
            raise ValueError("embedding_function must be callable")
        
        # Build filters
        filters = {}
        if role_filter:
            filters["role"] = role_filter
        
        # Search for similar vectors
        return self.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            filters=filters if filters else None,
            score_threshold=score_threshold
        )
    
    async def arecall_similar_messages(
        self,
        collection_name: str,
        query: str,
        embedding_function: callable,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        role_filter: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """Async version of recall_similar_messages."""
        # Generate query embedding
        import asyncio
        if asyncio.iscoroutinefunction(embedding_function):
            embedding_result = await embedding_function([query])
            query_embedding = embedding_result[0]
        else:
            query_embedding = embedding_function([query])[0]
        
        # Build filters
        filters = {}
        if role_filter:
            filters["role"] = role_filter
        
        # Search for similar vectors
        return await self.asearch(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            filters=filters if filters else None,
            score_threshold=score_threshold
        )
    
    def get_conversation_memory(
        self,
        collection_name: str,
        conversation_id: str,
        limit: int = 10
    ) -> List[VectorSearchResult]:
        """
        Retrieve all messages from a specific conversation.
        
        Args:
            collection_name: Collection to search in
            conversation_id: ID of the conversation
            limit: Maximum number of results
            
        Returns:
            List of messages from the conversation
        """
        # This would require a full scan with metadata filtering
        # Implementation depends on the specific vector store capabilities
        # For now, we'll use a placeholder that subclasses should override
        raise NotImplementedError("Subclasses should implement conversation-based retrieval")
    
    async def aget_conversation_memory(
        self,
        collection_name: str,
        conversation_id: str,
        limit: int = 10
    ) -> List[VectorSearchResult]:
        """Async version of get_conversation_memory."""
        raise NotImplementedError("Subclasses should implement conversation-based retrieval")
    
    # --- Utility Methods ---
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Collection to get stats for
            
        Returns:
            Dictionary with collection statistics (count, size, etc.)
        """
        # Default implementation returns basic info
        # Subclasses can override for more detailed stats
        return {
            "exists": self.collection_exists(collection_name),
            "name": collection_name
        }
    
    async def aget_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Async version of get_collection_stats."""
        return {
            "exists": await self.acollection_exists(collection_name),
            "name": collection_name
        }
    
    def cleanup(self) -> None:
        """Clean up any resources used by the vector store."""
        pass
    
    async def acleanup(self) -> None:
        """Async version of cleanup."""
        pass


# Convenience type alias for Message-based vector stores
MessageVectorStore = VectorStoreBase[Message]
