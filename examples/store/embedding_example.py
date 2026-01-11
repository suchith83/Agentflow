"""Example showing different embedding providers with stores."""

import asyncio

from agentflow.store import GoogleEmbedding, OpenAIEmbedding, QdrantStore
from agentflow.store.store_schema import MemoryType


async def example_openai_embedding():
    """Example: Using OpenAI embeddings."""
    print("=" * 60)
    print("Example 1: OpenAI Embeddings")
    print("=" * 60)

    # Create OpenAI embedding service
    embedding = OpenAIEmbedding(
        model="text-embedding-3-small"  # Default, 1536 dimensions
    )

    # Create Qdrant store with OpenAI embeddings
    store = QdrantStore(
        embedding=embedding,
        path="./memories_openai",
    )

    await store.asetup()

    print(f"✓ Store created with OpenAI embeddings")
    print(f"  Model: text-embedding-3-small")
    print(f"  Dimensions: {embedding.dimension}\n")

    await store.arelease()


async def example_google_embedding():
    """Example: Using Google embeddings."""
    print("=" * 60)
    print("Example 2: Google Embeddings")
    print("=" * 60)

    # Create Google embedding service
    embedding = GoogleEmbedding(
        model="text-embedding-004"  # Default, 768 dimensions
    )

    # Create Qdrant store with Google embeddings
    store = QdrantStore(
        embedding=embedding,
        path="./memories_google",
    )

    await store.asetup()

    print(f"✓ Store created with Google embeddings")
    print(f"  Model: text-embedding-004")
    print(f"  Dimensions: {embedding.dimension}\n")

    await store.arelease()


async def example_custom_dimensions():
    """Example: Custom embedding dimensions."""
    print("=" * 60)
    print("Example 2.5: Custom Dimensions")
    print("=" * 60)

    # Create Google embedding with custom dimensions
    embedding = GoogleEmbedding(
        model="text-embedding-004",
        output_dimensionality=256  # Custom dimension
    )

    store = QdrantStore(
        embedding=embedding,
        path="./memories_custom",
    )

    await store.asetup()

    print(f"✓ Store created with custom dimensions")
    print(f"  Model: text-embedding-004")
    print(f"  Dimensions: {embedding.dimension}\n")

    await store.arelease()


async def example_store_with_memory():
    """Example: Full workflow with embeddings."""
    print("=" * 60)
    print("Example 3: Full Store Workflow")
    print("=" * 60)

    # Use Google embeddings
    embedding = GoogleEmbedding()
    store = QdrantStore(
        embedding=embedding,
        path="./test_memories",
        default_collection="test_collection",
    )

    await store.asetup()

    config = {"user_id": "user1", "thread_id": "session1"}

    # Store memories
    memory_id = await store.astore(
        config=config,
        content="I love Python programming",
        memory_type=MemoryType.EPISODIC,
    )

    print(f"✓ Stored memory: {memory_id}")

    # Search memories
    results = await store.asearch(
        config=config,
        query="What do I love?",
        limit=5,
    )

    print(f"✓ Found {len(results)} results")
    for result in results:
        print(f"  - {result.content} (score: {result.score:.3f})")

    await store.arelease()
    print()


async def example_different_models():
    """Example: Different embedding models."""
    print("=" * 60)
    print("Example 4: Different Embedding Models")
    print("=" * 60)

    # OpenAI - Small model (fast, cheaper)
    openai_small = OpenAIEmbedding(model="text-embedding-3-small")
    print(f"✓ OpenAI small: {openai_small.dimension} dims")

    # OpenAI - Large model (better quality)
    openai_large = OpenAIEmbedding(model="text-embedding-3-large")
    print(f"✓ OpenAI large: {openai_large.dimension} dims")

    # Google - Default model
    google_default = GoogleEmbedding(model="text-embedding-004")
    print(f"✓ Google default: {google_default.dimension} dims")

    # Google - Custom dimensions
    google_custom = GoogleEmbedding(
        model="text-embedding-004",
        output_dimensionality=512
    )
    print(f"✓ Google custom: {google_custom.dimension} dims")

    print()


async def main():
    """Run all examples."""
    print("\n🚀 Embedding Provider Examples\n")

    await example_openai_embedding()
    await example_google_embedding()
    await example_custom_dimensions()
    await example_store_with_memory()
    await example_different_models()

    print("=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)

    print("\n📝 Available Providers:")
    print("  • OpenAIEmbedding - OpenAI's embedding models")
    print("  • GoogleEmbedding - Google's Gemini embedding models")
    print("  • Custom dimensions supported with GoogleEmbedding")


if __name__ == "__main__":
    asyncio.run(main())
