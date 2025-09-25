"""
Mem0 Store Implementation for PyAgenity Framework

This module provides a concrete implementation of BaseStore using Mem0
as the backend memory system. Mem0 provides high-level memory management
with automatic vector storage, semantic search, and conversation tracking.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from .base_store import BaseStore, MemorySearchResult, MemoryRecord
from pyagenity.utils.message import Message

try:
    from mem0 import Memory
except ImportError:
    raise ImportError(
        "Mem0 not installed. Install with: pip install mem0ai"
    )

"""
Mem0 Store Implementation for PyAgenity Framework

This module provides a concrete implementation of BaseStore using Mem0
as the backend memory system. Mem0 provides high-level memory management
with automatic vector storage, semantic search, and conversation tracking.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from .base_store import BaseStore, MemorySearchResult, MemoryRecord
from pyagenity.utils.message import Message

try:
    from mem0 import Memory
except ImportError:
    raise ImportError(
        "Mem0 not installed. Install with: pip install mem0ai"
    )

logger = logging.getLogger(__name__)


class Mem0Store(BaseStore):
    """
    Mem0-based implementation of BaseStore for long-term memory.
    
    This store provides a high-level interface to Mem0's memory management
    system, which handles vector storage, embeddings, and semantic search
    automatically.
    
    Features:
    - Automatic embedding generation via Mem0
    - User and agent-centric memory management
    - Semantic search with relevance scoring
    - Memory lifecycle management (CRUD operations)
    - Conversation context tracking
    """
    
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        **kwargs
    ):
        """
        Initialize Mem0Store.
        
        Args:
            config: Mem0 configuration dict (llm, embedder, vector_store)
            user_id: Default user ID for memory operations
            agent_id: Default agent ID for memory operations
            app_id: Application ID for memory context
            **kwargs: Additional arguments passed to BaseStore
        """
        super().__init__(**kwargs)
        
        self.config = config or {}
        self.default_user_id = user_id or "default_user"
        self.default_agent_id = agent_id
        self.app_id = app_id or "pyagenity_app"
        
        # Track memory_id to user_id mapping for retrieval
        self._memory_user_map: dict[str, str] = {}
        
        # Initialize Mem0
        try:
            if config:
                self.memory = Memory.from_config(config)
            else:
                self.memory = Memory()
        except Exception as e:
            logger.error(f"Failed to initialize Mem0 Memory: {e}")
            raise RuntimeError(f"Failed to initialize Mem0 Memory: {e}")
        
        logger.info(
            f"Initialized Mem0Store for user: {self.default_user_id}, "
            f"agent: {self.default_agent_id}, app: {self.app_id}"
        )
    
    # --- Core Memory Operations (BaseStore Interface) ---
    
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
        Add a new memory to Mem0.
        
        Args:
            content: The memory content
            user_id: User identifier (defaults to instance default)
            agent_id: Agent identifier (defaults to instance default)
            memory_type: Type of memory (episodic, semantic, etc.)
            category: Memory category for organization
            metadata: Additional metadata
            **kwargs: Store-specific parameters
            
        Returns:
            Memory ID
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        
        # Use provided IDs or defaults
        effective_user_id = user_id or self.default_user_id
        effective_agent_id = agent_id or self.default_agent_id
        
        # Generate memory ID
        memory_id = str(uuid4())
        
        # Build comprehensive metadata
        mem0_metadata = {
            "memory_id": memory_id,
            "agent_id": effective_agent_id,
            "memory_type": memory_type,
            "category": category,
            "app_id": self.app_id,
            "created_at": datetime.now().isoformat(),
            **(metadata or {}),
        }
        
        try:
            # Add memory to Mem0
            result = self.memory.add(
                messages=[{"role": "user", "content": content}],
                user_id=effective_user_id,
                metadata=mem0_metadata,
                **kwargs
            )
            
            # Extract the actual Mem0 ID if available
            mem0_id = None
            if isinstance(result, dict):
                if "results" in result and result["results"]:
                    mem0_id = result["results"][0].get("id")
                elif "id" in result:
                    mem0_id = result["id"]
            
            # Update metadata with Mem0 ID if available
            if mem0_id:
                mem0_metadata["mem0_id"] = str(mem0_id)
                # Re-add with updated metadata (Mem0 limitation workaround)
                logger.debug(f"Mem0 assigned ID: {mem0_id} for memory: {memory_id}")
            
            # Track the memory_id to user_id mapping for retrieval
            self._memory_user_map[memory_id] = effective_user_id
            
            logger.info(f"Added memory {memory_id} for user {effective_user_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise RuntimeError(f"Failed to add memory: {e}")
    
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
        # Mem0 doesn't have native async support, run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.add(content, user_id, agent_id, memory_type, category, metadata, **kwargs)
        )
    
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
            user_id: Filter by user (defaults to instance default)
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
        effective_user_id = user_id or self.default_user_id
        
        # Build filters for Mem0
        # Note: Mem0 with Qdrant requires indexed fields for filtering
        # memory_type and app_id filtering cause index errors, so we filter in post-processing
        mem0_filters = {}
        if agent_id:
            mem0_filters["agent_id"] = agent_id
        # Skip memory_type and app_id filters to avoid Qdrant index errors
        # if memory_type:
        #     mem0_filters["memory_type"] = memory_type
        if category:
            mem0_filters["category"] = category
        # Skip app_id filter to avoid Qdrant index errors  
        # if self.app_id:
        #     mem0_filters["app_id"] = self.app_id
        if filters:
            mem0_filters.update(filters)
        
        try:
            # Search in Mem0
            results = self.memory.search(
                query=query,
                user_id=effective_user_id,
                limit=limit,
                filters=mem0_filters if mem0_filters else None,
                **kwargs
            )
            
            search_results = []
            
            # Process Mem0 results
            if isinstance(results, dict) and "results" in results:
                for result in results["results"]:
                    score = result.get("score", 0.0)

                    
                    # Apply score threshold
                    if score_threshold is not None and score < score_threshold:
                        continue

                    
                    # Extract memory data
                    memory_content = result.get("memory", "")
                    result_metadata = result.get("metadata", {})
                    mem0_id = result.get("id")
                    
                    # Apply memory_type, app_id, and additional filters in post-processing
                    if memory_type and result_metadata.get("memory_type") != memory_type:
                        continue
                    if self.app_id and result_metadata.get("app_id") != self.app_id:
                        continue
                    
                    # Apply additional filters from the filters dict
                    if filters:
                        skip_result = False
                        for filter_key, filter_value in filters.items():
                            metadata_value = result_metadata.get(filter_key)
                            if metadata_value != filter_value:
                                skip_result = True
                                break
                        if skip_result:
                            print("broke t filters")
                            continue
                    # Create MemorySearchResult
                    search_results.append(MemorySearchResult(
                        id=result_metadata.get("memory_id", str(mem0_id) if mem0_id else str(uuid4())),
                        content=memory_content,
                        score=float(score),
                        memory_type=result_metadata.get("memory_type", "episodic"),
                        metadata=result_metadata,
                        user_id=effective_user_id,
                        agent_id=result_metadata.get("agent_id"),
                        created_at=self._parse_datetime(result_metadata.get("created_at")),
                        updated_at=self._parse_datetime(result_metadata.get("updated_at")),
                    ))
            
            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Search failed: {e}")
    
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(
                query, user_id, agent_id, memory_type, category, 
                limit, score_threshold, filters, **kwargs
            )
        )
    
    def get(self, memory_id: str, **kwargs) -> MemorySearchResult | None:
        """
        Get a specific memory by ID.
        
        Note: Mem0 doesn't provide direct ID-based retrieval,
        so we get all memories and filter by memory_id in metadata.
        """
        try:
            # First, check if we know which user this memory belongs to
            target_user_id = self._memory_user_map.get(memory_id, self.default_user_id)
            
            # Get all memories for the target user
            all_memories = self.memory.get_all(user_id=target_user_id)
            
            # Check memories for this specific memory_id
            for memory in all_memories:
                metadata = memory.get("metadata", {})
                if metadata.get("memory_id") == memory_id:
                    return MemorySearchResult(
                        id=memory_id,
                        content=memory.get("memory", ""),
                        score=1.0,  # Perfect match
                        memory_type=metadata.get("memory_type", "episodic"),
                        metadata=metadata,
                        user_id=memory.get("user_id", target_user_id),
                        agent_id=metadata.get("agent_id"),
                        created_at=self._parse_datetime(metadata.get("created_at")),
                        updated_at=self._parse_datetime(metadata.get("updated_at")),
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Get operation failed: {e}")
            return None
    
    async def aget(self, memory_id: str, **kwargs) -> MemorySearchResult | None:
        """Async version of get."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get(memory_id, **kwargs))
    
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Update an existing memory.
        
        Note: Mem0's update capabilities are limited. We use the memory.update
        method if available, otherwise fall back to get + delete + add pattern.
        """
        try:
            # Try to get the existing memory first
            existing = self.get(memory_id)
            if not existing:
                raise ValueError(f"Memory with ID {memory_id} not found")
            
            # Try using Mem0's update method if available
            mem0_id = existing.metadata.get("mem0_id")
            if hasattr(self.memory, 'update') and mem0_id:
                update_data = {}
                if content:
                    update_data["data"] = content
                if metadata:
                    # Merge with existing metadata
                    updated_metadata = {**existing.metadata, **metadata}
                    updated_metadata["updated_at"] = datetime.now().isoformat()
                    update_data["metadata"] = updated_metadata
                
                if update_data:
                    self.memory.update(memory_id=mem0_id, **update_data)
                    logger.info(f"Updated memory {memory_id}")
                    return
            
            # Fallback: manual update by recreating
            logger.warning("Using fallback update method (delete + add)")
            
            # Prepare updated content and metadata
            updated_content = content if content is not None else existing.content
            updated_metadata = {**existing.metadata}
            if metadata:
                updated_metadata.update(metadata)
            updated_metadata["updated_at"] = datetime.now().isoformat()
            
            # Delete old memory (if we have mem0_id)
            if mem0_id and hasattr(self.memory, 'delete'):
                try:
                    self.memory.delete(memory_id=mem0_id)
                except Exception as e:
                    logger.warning(f"Failed to delete old memory during update: {e}")
            
            # Add updated memory
            self.add(
                content=updated_content,
                user_id=existing.user_id,
                agent_id=existing.agent_id,
                memory_type=existing.memory_type,
                category=updated_metadata.get("category", "general"),
                metadata=updated_metadata,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Update failed: {e}")
            raise RuntimeError(f"Update failed: {e}")
    
    async def aupdate(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Async version of update."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self.update(memory_id, content, metadata, **kwargs)
        )
    
    def delete(self, memory_id: str, **kwargs) -> None:
        """Delete a memory by ID."""
        try:
            # Get the memory to find its Mem0 ID
            existing = self.get(memory_id)
            if not existing:
                logger.warning(f"Memory {memory_id} not found for deletion")
                return
            
            mem0_id = existing.metadata.get("mem0_id")
            if mem0_id and hasattr(self.memory, 'delete'):
                self.memory.delete(memory_id=mem0_id)
                logger.info(f"Deleted memory {memory_id} (Mem0 ID: {mem0_id})")
            else:
                logger.warning(
                    f"Cannot delete memory {memory_id}: no Mem0 ID or delete method unavailable"
                )
            
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise RuntimeError(f"Delete failed: {e}")
    
    async def adelete(self, memory_id: str, **kwargs) -> None:
        """Async version of delete."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.delete(memory_id, **kwargs))
    
    # --- User/Agent Management Overrides ---
    
    def get_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        category: str | None = None,
        limit: int = 100,
        **kwargs,
    ) -> list[MemorySearchResult]:
        """Get all memories for a user/agent with optional filters."""
        effective_user_id = user_id or self.default_user_id
        
        try:
            # Get all memories from Mem0
            all_memories_response = self.memory.get_all(user_id=effective_user_id)
            
            # Handle different response formats from Mem0
            all_memories = []
            if isinstance(all_memories_response, list):
                all_memories = all_memories_response
            elif isinstance(all_memories_response, dict) and "results" in all_memories_response:
                all_memories = all_memories_response["results"]
            else:
                logger.warning(f"Unexpected response format from Mem0.get_all(): {type(all_memories_response)}")
                return []
            
            results = []
            for memory in all_memories:
                # Handle case where memory might be a string or dict
                if isinstance(memory, str):
                    # If memory is just a string, create a minimal structure
                    memory = {"memory": memory, "metadata": {}}
                elif not isinstance(memory, dict):
                    logger.warning(f"Unexpected memory format: {type(memory)}")
                    continue
                
                metadata = memory.get("metadata", {})
                
                # Apply filters
                if agent_id and metadata.get("agent_id") != agent_id:
                    continue
                if memory_type and metadata.get("memory_type") != memory_type:
                    continue
                if category and metadata.get("category") != category:
                    continue
                if self.app_id and metadata.get("app_id") != self.app_id:
                    continue
                
                results.append(MemorySearchResult(
                    id=metadata.get("memory_id", str(uuid4())),
                    content=memory.get("memory", ""),
                    score=1.0,
                    memory_type=metadata.get("memory_type", "episodic"),
                    metadata=metadata,
                    user_id=effective_user_id,
                    agent_id=metadata.get("agent_id"),
                    created_at=self._parse_datetime(metadata.get("created_at")),
                    updated_at=self._parse_datetime(metadata.get("updated_at")),
                ))
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []
    
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.get_all(user_id, agent_id, memory_type, category, limit, **kwargs)
        )
    
    def delete_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        category: str | None = None,
        **kwargs,
    ) -> int:
        """Delete all memories matching criteria."""
        try:
            # Get matching memories
            memories_to_delete = self.get_all(
                user_id=user_id,
                agent_id=agent_id,
                memory_type=memory_type,
                category=category,
                limit=10000,  # Large limit to get all
                **kwargs
            )
            
            count = 0
            for memory in memories_to_delete:
                try:
                    self.delete(memory.id)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete memory {memory.id}: {e}")
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to delete all memories: {e}")
            return 0
    
    async def adelete_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        category: str | None = None,
        **kwargs,
    ) -> int:
        """Async version of delete_all."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.delete_all(user_id, agent_id, memory_type, category, **kwargs)
        )
    
    # --- Message-Specific Convenience Methods (Override BaseStore) ---

    def store_message(
        self,
        message: Message,
        user_id: str | None = None,
        agent_id: str | None = None,
        additional_metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Store a PyAgenity Message as memory using direct add method to avoid parameter conflicts."""
        # Extract message content and role (using .text() method for proper content extraction)
        content = message.text()
        role = message.role
        
        # Build metadata with message info
        message_metadata = {
            "role": role,
            "message_id": message.message_id,
            "timestamp": message.timestamp.isoformat() if message.timestamp else datetime.now().isoformat(),
            **(additional_metadata or {}),
        }
        
        # Use memory_type from kwargs or default to "episodic" for messages (to match BaseStore)
        memory_type = kwargs.pop("memory_type", "episodic")
        category = kwargs.pop("category", "message")
        
        # Call add method directly to avoid parameter conflicts
        return self.add(
            content=content,
            user_id=user_id,
            agent_id=agent_id,
            memory_type=memory_type,
            category=category,
            metadata=message_metadata,
            **kwargs
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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.store_message(message, user_id, agent_id, additional_metadata, **kwargs)
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
        """Retrieve messages similar to the query with role filtering support."""
        # Build filters for role filtering
        filters = {}
        if role_filter:
            filters["role"] = role_filter

        # Search using the main search method with episodic memory type for messages
        return self.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            memory_type="episodic",  # Use "episodic" for stored messages to match BaseStore
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
        # Build filters for role filtering
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

    # --- Utility Methods ---
    
    def _parse_datetime(self, timestamp_str: str | None) -> datetime | None:
        """Parse ISO timestamp string to datetime object."""
        if not timestamp_str:
            return None
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
    
    def get_stats(self, user_id: str | None = None, agent_id: str | None = None) -> dict[str, Any]:
        """Get statistics about stored memories."""
        effective_user_id = user_id or self.default_user_id
        
        try:
            memories = self.get_all(user_id=effective_user_id, agent_id=agent_id, limit=1000)
            
            # Count by memory type and category
            memory_types = {}
            categories = {}
            
            for memory in memories:
                memory_type = memory.memory_type
                category = memory.metadata.get("category", "general")
                
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                categories[category] = categories.get(category, 0) + 1
            
            return {
                "total_memories": len(memories),
                "user_id": effective_user_id,
                "agent_id": agent_id,
                "app_id": self.app_id,
                "memory_types": memory_types,
                "categories": categories,
                "memory_types_list": list(memory_types.keys()),
                "categories_list": list(categories.keys()),
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    async def aget_stats(
        self, user_id: str | None = None, agent_id: str | None = None
    ) -> dict[str, Any]:
        """Async version of get_stats."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.get_stats(user_id, agent_id))
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Mem0 doesn't require explicit cleanup, but we can log
            logger.info("Mem0Store cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    async def acleanup(self) -> None:
        """Async version of cleanup."""
        self.cleanup()


# Convenience factory functions

def create_mem0_store(
    config: dict[str, Any] | None = None,
    user_id: str = "default_user",
    agent_id: str | None = None,
    app_id: str = "pyagenity_app"
) -> Mem0Store:
    """
    Create a Mem0Store with the given configuration.
    
    Args:
        config: Mem0 configuration dict
        user_id: User identifier for memory isolation
        agent_id: Agent identifier
        app_id: Application identifier
        
    Returns:
        Configured Mem0Store instance
    """
    return Mem0Store(config=config, user_id=user_id, agent_id=agent_id, app_id=app_id)


def create_mem0_store_with_qdrant(
    qdrant_url: str,
    qdrant_api_key: str | None = None,
    collection_name: str = "pyagenity_memories",
    embedding_model: str = "text-embedding-ada-002",
    llm_model: str = "gpt-4o-mini",
    user_id: str = "default_user",
    agent_id: str | None = None,
    app_id: str = "pyagenity_app",
    **kwargs
) -> Mem0Store:
    """
    Create a Mem0Store configured with Qdrant backend.
    
    Args:
        qdrant_url: Qdrant server URL
        qdrant_api_key: Qdrant API key (for cloud)
        collection_name: Qdrant collection name
        embedding_model: Embedding model to use
        llm_model: LLM model for processing
        user_id: User identifier
        agent_id: Agent identifier
        app_id: Application identifier
        **kwargs: Additional configuration options including:
            - embedder_provider: Provider for embeddings (default: "openai")
            - llm_provider: Provider for LLM (default: "openai")
            - vector_store_config: Additional vector store config
            - embedder_config: Additional embedder config
            - llm_config: Additional LLM config
        
    Returns:
        Mem0Store configured with Qdrant
    """
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection_name,
                "url": qdrant_url,
                "api_key": qdrant_api_key,
                **kwargs.get("vector_store_config", {})
            }
        },
        "embedder": {
            "provider": kwargs.get("embedder_provider", "openai"),
            "config": {
                "model": embedding_model,
                **kwargs.get("embedder_config", {})
            }
        },
        "llm": {
            "provider": kwargs.get("llm_provider", "openai"),
            "config": {
                "model": llm_model,
                **kwargs.get("llm_config", {})
            }
        }
    }

    print("config: ", config)
    
    return Mem0Store(config=config, user_id=user_id, agent_id=agent_id, app_id=app_id)
