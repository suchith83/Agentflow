from .base_store import BaseStore, DistanceMetric, MemoryRecord, MemorySearchResult


try:
    from .qdrant_store import (
        QdrantVectorStore,
        create_cloud_qdrant_vector_store,
        create_local_qdrant_vector_store,
        create_remote_qdrant_vector_store,
    )

    __all__ = [
        "BaseStore",
        "DistanceMetric",
        "MemorySearchResult",
        "MemoryRecord",
        "QdrantVectorStore",
        "create_local_qdrant_vector_store",
        "create_remote_qdrant_vector_store",
        "create_cloud_qdrant_vector_store",
    ]
except ImportError:
    # qdrant-client not installed
    __all__ = [
        "BaseStore",
        "DistanceMetric",
        "MemorySearchResult",
        "MemoryRecord",
    ]

# Try to import Mem0Store (optional dependency)
try:
    from .mem0_store import (  # noqa: F401
        Mem0Store,
        create_mem0_store,
        create_mem0_store_with_qdrant,
    )

    __all__.extend(["Mem0Store", "create_mem0_store", "create_mem0_store_with_qdrant"])
except ImportError:
    # mem0 not installed
    pass
