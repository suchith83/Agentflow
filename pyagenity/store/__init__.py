from .base_store import BaseStore
from .store_schema import DistanceMetric, MemoryRecord, MemorySearchResult, MemoryType


try:
    from .qdrant_store import (
        EmbeddingService,
        OpenAIEmbeddingService,
        QdrantStore,
        create_cloud_qdrant_store,
        create_local_qdrant_store,
        create_remote_qdrant_store,
    )

    __all__ = [
        "BaseStore",
        "DistanceMetric",
        "MemorySearchResult",
        "MemoryRecord",
        "MemoryType",
        "EmbeddingService",
        "OpenAIEmbeddingService",
        "QdrantStore",
        "create_local_qdrant_store",
        "create_remote_qdrant_store",
        "create_cloud_qdrant_store",
    ]
except ImportError:
    # qdrant-client not installed
    __all__ = [
        "BaseStore",
        "DistanceMetric",
        "MemorySearchResult",
        "MemoryRecord",
        "MemoryType",
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
