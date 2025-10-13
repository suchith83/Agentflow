#!/usr/bin/env python3
"""
QdrantStore Usage Example

This example demonstrates how to use the QdrantStore implementation
with different embedding services and configurations.
"""

import asyncio
import os
from typing import Any

from agentflow.state import Message
from agentflow.store import MemoryType, OpenAIEmbedding, QdrantStore
from agentflow.store.embedding.base_embedding import BaseEmbedding


# Example 1: Simple embedding service for testing
class SimpleEmbeddingService(BaseEmbedding):
    """Simple embedding service that creates mock embeddings."""

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate simple embeddings based on text hashes."""
        return [await self.aembed(text) for text in texts]

    async def aembed(self, text: str) -> list[float]:
        """Generate a simple embedding based on text hash."""
        text_hash = hash(text)
        return [float((text_hash + i) % 1000) / 1000.0 for i in range(self._dimension)]

    @property
    def dimension(self) -> int:
        return self._dimension


async def basic_usage_example():
    """Basic usage example with local Qdrant."""
    print("=== Basic QdrantStore Usage Example ===\n")

    # Create embedding service
    embedding_service = SimpleEmbeddingService(dimension=384)

    # Create local Qdrant store
    store = QdrantStore(
        embedding=embedding_service,
        path="./example_qdrant_data",  # Local file-based storage
        default_collection="example_memories",
    )

    try:
        # Setup the store
        await store.asetup()
        print("✓ Store initialized successfully")

        # Configuration for this session
        config = {
            "user_id": "demo_user",
            "agent_id": "demo_agent",
            "collection": "example_memories",
        }

        # Store some memories
        print("\n--- Storing Memories ---")

        memory1_id = await store.astore(
            config=config,
            content="I love Python programming",
            memory_type=MemoryType.EPISODIC,
            category="preferences",
            metadata={"topic": "programming", "sentiment": "positive"},
        )
        print(f"✓ Stored memory 1: {memory1_id[:8]}...")

        memory2_id = await store.astore(
            config=config,
            content="Remember to buy groceries tomorrow",
            memory_type=MemoryType.PROCEDURAL,
            category="tasks",
            metadata={"priority": "high", "due_date": "2024-01-15"},
        )
        print(f"✓ Stored memory 2: {memory2_id[:8]}...")

        # Create a Message object and store it
        message = Message.text_message("The weather is nice today", role="user")
        memory3_id = await store.astore(
            config=config, content=message, memory_type=MemoryType.EPISODIC, category="observations"
        )
        print(f"✓ Stored memory 3 (Message): {memory3_id[:8]}...")

        # Search for memories
        print("\n--- Searching Memories ---")

        # Search for programming-related memories
        programming_results = await store.asearch(
            config=config, query="Python programming language", limit=3, score_threshold=0.1
        )
        print(f"✓ Found {len(programming_results)} programming-related memories")
        for result in programming_results:
            print(f"  - {result.content} (score: {result.score:.3f})")

        # Search for task-related memories
        task_results = await store.asearch(
            config=config, query="things to do tomorrow", memory_type=MemoryType.PROCEDURAL, limit=3
        )
        print(f"✓ Found {len(task_results)} task-related memories")
        for result in task_results:
            print(f"  - {result.content} (score: {result.score:.3f})")

        # Retrieve a specific memory
        print("\n--- Retrieving Specific Memory ---")
        specific_memory = await store.aget(config=config, memory_id=memory1_id)
        if specific_memory:
            print(f"✓ Retrieved memory: {specific_memory.content}")
            print(f"  Metadata: {specific_memory.metadata}")

        # Update a memory
        print("\n--- Updating Memory ---")
        await store.aupdate(
            config=config,
            memory_id=memory2_id,
            content="Remember to buy groceries and coffee tomorrow",
            metadata={"priority": "high", "due_date": "2024-01-15", "updated": True},
        )
        print(f"✓ Updated memory {memory2_id[:8]}...")

        # Verify the update
        updated_memory = await store.aget(config=config, memory_id=memory2_id)
        if updated_memory:
            print(f"  Updated content: {updated_memory.content}")

        # Batch store example
        print("\n--- Batch Storage ---")
        batch_memories = [
            "I learned about machine learning today",
            "The meeting went well this morning",
            "Need to review the project proposal",
        ]

        batch_id = await store.store(
            config=config,
            content=batch_memories,
            memory_type=MemoryType.EPISODIC,
            category="daily_logs",
        )
        print(f"✓ Batch stored {len(batch_memories)} memories: {batch_id}")

        # Clean up one memory
        print("\n--- Deleting Memory ---")
        await store.adelete(config=config, memory_id=memory3_id)
        print(f"✓ Deleted memory {memory3_id[:8]}...")

        # Verify deletion
        deleted_memory = await store.aget(config=config, memory_id=memory3_id)
        print(f"✓ Memory deletion confirmed: {deleted_memory is None}")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Clean up resources
        await store.arelease()
        print("\n✓ Store resources released")


async def openai_embedding_example():
    """Example using OpenAI embeddings (requires API key)."""
    print("\n=== OpenAI Embedding Example ===\n")

    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping OpenAI example.")
        print("   Set your API key with: export OPENAI_API_KEY=your_key_here")
        return

    try:
        # Create OpenAI embedding service
        embedding_service = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=openai_api_key,
        )

        # Create cloud Qdrant store (you would need real Qdrant cloud credentials)
        store = QdrantStore(
            embedding=embedding_service,
            path="./openai_qdrant_data",  # Using local for demo
        )

        await store.asetup()
        print("✓ OpenAI-powered store initialized")

        config = {"user_id": "openai_user", "agent_id": "smart_agent"}

        # Store a memory with high-quality embeddings
        memory_id = await store.astore(
            config=config,
            content="Artificial intelligence is transforming how we work and live",
            memory_type=MemoryType.SEMANTIC,
            category="ai_insights",
        )
        print(f"✓ Stored memory with OpenAI embeddings: {memory_id[:8]}...")

        # Search with semantic understanding
        results = await store.asearch(
            config=config, query="machine learning impact on society", limit=5
        )
        print(f"✓ Found {len(results)} semantically related memories")

        await store.arelease()

    except ImportError:
        print("❌ OpenAI package not installed. Install with: pip install openai")
    except Exception as e:
        print(f"❌ Error with OpenAI embeddings: {e}")


def remote_qdrant_example():
    """Example configuration for remote Qdrant server."""
    print("\n=== Remote Qdrant Configuration Example ===\n")

    # This is just configuration - not actually connecting

    from agentflow.store.qdrant_store import create_remote_qdrant_store

    print("Example configurations for different Qdrant deployments:")

    print("\n1. Local Qdrant server:")
    print("""
    store = create_remote_qdrant_store(
        host="localhost",
        port=6333,
        embedding_service=your_embedding_service
    )
    """)

    print("2. Remote Qdrant server:")
    print("""
    store = create_remote_qdrant_store(
        host="your-qdrant-server.com",
        port=6333,
        embedding_service=your_embedding_service
    )
    """)

    print("3. Qdrant Cloud:")
    print("""
    from agentflow.store.qdrant_store import create_cloud_qdrant_store
    
    store = create_cloud_qdrant_store(
        url="https://your-cluster.qdrant.io",
        api_key="your-qdrant-api-key",
        embedding_service=your_embedding_service
    )
    """)

    print("4. With custom configuration:")
    print("""
    store = QdrantStore(
        embedding_service=your_embedding_service,
        host="localhost",
        port=6333,
        default_collection="my_memories",
        distance_metric=DistanceMetric.EUCLIDEAN,
        timeout=30,  # Custom Qdrant client parameters
        prefer_grpc=True
    )
    """)


async def main():
    """Run all examples."""
    print("🚀 QdrantStore Examples\n")

    # Run basic example
    await basic_usage_example()

    # Run OpenAI example if API key is available
    await openai_embedding_example()

    # Show configuration examples
    remote_qdrant_example()

    print("\n🎉 Examples completed!")
    print("\nNext steps:")
    print("1. Install qdrant-client: pip install 'agentflow[qdrant]'")
    print("2. For OpenAI embeddings: pip install openai")
    print("3. Start local Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    print("4. Set OPENAI_API_KEY environment variable for OpenAI embeddings")


if __name__ == "__main__":
    asyncio.run(main())
