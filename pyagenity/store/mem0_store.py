"""
Simplified Mem0 Store Implementation for PyAgenity Framework

This module provides a clean, async-first implementation of BaseStore using Mem0
as the backend memory system. Mem0 provides high-level memory management
with automatic vector storage, semantic search, and conversation tracking.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from .base_store import BaseStore, MemorySearchResult


try:
    from mem0 import Memory
except ImportError:
    raise ImportError("Mem0 not installed. Install with: pip install mem0ai")

logger = logging.getLogger(__name__)


class Mem0Store(BaseStore):
    """
    Simplified Mem0-based implementation of BaseStore for long-term memory.

    This store provides a high-level interface to Mem0's memory management
    system, which handles vector storage, embeddings, and semantic search
    automatically.

    Features:
    - Automatic embedding generation via Mem0
    - User and agent-centric memory management
    - Semantic search with relevance scoring
    - Async-first design for better performance
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        **kwargs,
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
            self.memory = Memory(config=self.config)
            logger.info("Successfully initialized Mem0 client")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {e}")
            raise

        logger.info(
            f"Initialized Mem0Store for user: {self.default_user_id}, "
            f"agent: {self.default_agent_id}, app: {self.app_id}"
        )

    # --- Core Memory Operations (BaseStore Interface) ---

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
        """Add a new memory to Mem0."""
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
            # Run in thread pool since Mem0 doesn't have native async support
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.memory.add(
                    content,
                    user_id=effective_user_id,
                    metadata=mem0_metadata,
                ),
            )

            # Store mapping for retrieval
            self._memory_user_map[memory_id] = effective_user_id

            logger.debug(f"Added memory {memory_id} for user {effective_user_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise

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
        """Search memories by content similarity."""
        effective_user_id = user_id or self.default_user_id

        try:
            # Run search in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.memory.search(
                    query=query or "",
                    user_id=effective_user_id,
                    limit=limit,
                ),
            )

            # Convert and filter results
            memory_results = []
            for result in results:
                # Apply post-processing filters
                result_metadata = result.get("metadata", {})
                
                # Apply filters
                if agent_id and result_metadata.get("agent_id") != agent_id:
                    continue
                if memory_type and result_metadata.get("memory_type") != memory_type:
                    continue
                if category and result_metadata.get("category") != category:
                    continue
                if self.app_id and result_metadata.get("app_id") != self.app_id:
                    continue
                if score_threshold and result.get("score", 0.0) < score_threshold:
                    continue

                # Apply additional filters
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if result_metadata.get(key) != value:
                            skip = True
                            break
                    if skip:
                        continue

                # Create search result
                memory_results.append(self._create_search_result(result, effective_user_id))

            return memory_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get(self, memory_id: str, **kwargs) -> MemorySearchResult | None:
        """Get a specific memory by ID."""
        try:
            # Get all memories and filter by memory_id in metadata
            # (Mem0 doesn't provide direct ID-based retrieval)
            loop = asyncio.get_event_loop()
            
            # Try to get user_id from mapping or use default
            user_id = self._memory_user_map.get(memory_id, self.default_user_id)
            
            all_memories = await loop.run_in_executor(
                None, lambda: self.memory.get_all(user_id=user_id)
            )

            # Find the specific memory
            for memory_dict in self._parse_mem0_response(all_memories):
                metadata = memory_dict.get("metadata", {})
                if metadata.get("memory_id") == memory_id:
                    return self._create_search_result(memory_dict, user_id)

            return None

        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Update an existing memory."""
        try:
            # Get existing memory first
            existing = await self.get(memory_id)
            if not existing:
                raise ValueError(f"Memory {memory_id} not found")

            # Get user_id from existing memory or mapping
            user_id = existing.user_id or self._memory_user_map.get(memory_id, self.default_user_id)

            # Update content if provided
            if content is not None:
                updated_metadata = existing.metadata.copy() if existing.metadata else {}
                updated_metadata["updated_at"] = datetime.now().isoformat()
                if metadata:
                    updated_metadata.update(metadata)

                # Run update in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.memory.update(
                        memory_id, content, user_id=user_id, metadata=updated_metadata
                    ),
                )
            
            logger.debug(f"Updated memory {memory_id}")

        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            raise

    async def delete(self, memory_id: str, **kwargs) -> None:
        """Delete a memory by ID."""
        try:
            # Get user_id from mapping or use default
            user_id = self._memory_user_map.get(memory_id, self.default_user_id)

            # Run delete in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self.memory.delete(memory_id, user_id=user_id)
            )

            # Clean up mapping
            self._memory_user_map.pop(memory_id, None)
            
            logger.debug(f"Deleted memory {memory_id}")

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise

    # --- Utility Methods ---

    def _create_search_result(
        self, result: dict[str, Any], effective_user_id: str
    ) -> MemorySearchResult:
        """Create a MemorySearchResult from a Mem0 result."""
        memory_content = result.get("memory", "")
        result_metadata = result.get("metadata", {})
        mem0_id = result.get("id")
        score = result.get("score", 0.0)

        return MemorySearchResult(
            id=result_metadata.get("memory_id", str(mem0_id) if mem0_id else str(uuid4())),
            content=memory_content,
            score=float(score),
            memory_type=result_metadata.get("memory_type", "episodic"),
            metadata=result_metadata,
            user_id=effective_user_id,
            agent_id=result_metadata.get("agent_id"),
            created_at=self._parse_datetime(result_metadata.get("created_at")),
            updated_at=self._parse_datetime(result_metadata.get("updated_at")),
        )

    def _parse_mem0_response(self, response: Any) -> list[dict]:
        """Parse Mem0 response to extract memories list."""
        if isinstance(response, list):
            return [self._normalize_memory_dict(m) for m in response if self._normalize_memory_dict(m)]
        if isinstance(response, dict) and "results" in response:
            return [self._normalize_memory_dict(m) for m in response["results"] if self._normalize_memory_dict(m)]

        logger.warning(f"Unexpected response format from Mem0.get_all(): {type(response)}")
        return []

    def _normalize_memory_dict(self, memory: Any) -> dict | None:
        """Normalize memory data to dict format."""
        if isinstance(memory, str):
            return {"memory": memory, "metadata": {}}
        if isinstance(memory, dict):
            return memory

        logger.warning(f"Unexpected memory format: {type(memory)}")
        return None

    def _parse_datetime(self, timestamp_str: str | None) -> datetime | None:
        """Parse datetime string to datetime object."""
        if not timestamp_str:
            return None
        try:
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            return None

    async def get_stats(self, user_id: str | None = None, agent_id: str | None = None) -> dict[str, Any]:
        """Get statistics about stored memories."""
        effective_user_id = user_id or self.default_user_id
        
        try:
            loop = asyncio.get_event_loop()
            all_memories = await loop.run_in_executor(
                None, lambda: self.memory.get_all(user_id=effective_user_id)
            )

            memories = self._parse_mem0_response(all_memories)
            
            # Apply agent filter if specified
            if agent_id:
                memories = [m for m in memories if m.get("metadata", {}).get("agent_id") == agent_id]
            
            # Apply app filter
            if self.app_id:
                memories = [m for m in memories if m.get("metadata", {}).get("app_id") == self.app_id]

            # Calculate stats
            memory_types = {}
            categories = {}
            for memory in memories:
                metadata = memory.get("metadata", {})
                mem_type = metadata.get("memory_type", "episodic")
                category = metadata.get("category", "general")
                
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
                categories[category] = categories.get(category, 0) + 1

            return {
                "total_memories": len(memories),
                "user_id": effective_user_id,
                "agent_id": agent_id,
                "app_id": self.app_id,
                "memory_types": memory_types,
                "categories": categories,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_memories": 0,
                "user_id": effective_user_id,
                "agent_id": agent_id,
                "app_id": self.app_id,
                "error": str(e),
            }

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Clear memory mapping
            self._memory_user_map.clear()
            logger.info("Mem0Store cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Convenience factory functions


def create_mem0_store(
    config: dict[str, Any] | None = None,
    user_id: str = "default_user",
    agent_id: str | None = None,
    app_id: str = "pyagenity_app",
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
    **kwargs,
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
        **kwargs: Additional configuration options

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
                **kwargs.get("vector_store_config", {}),
            },
        },
        "embedder": {
            "provider": kwargs.get("embedder_provider", "openai"),
            "config": {"model": embedding_model, **kwargs.get("embedder_config", {})},
        },
        "llm": {
            "provider": kwargs.get("llm_provider", "openai"),
            "config": {"model": llm_model, **kwargs.get("llm_config", {})},
        },
    }

    return Mem0Store(config=config, user_id=user_id, agent_id=agent_id, app_id=app_id)
