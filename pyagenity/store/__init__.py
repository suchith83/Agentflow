from .base_store import BaseStore
from .embedding import BaseEmbedding, OpenAIEmbedding
from .store_schema import DistanceMetric, MemoryRecord, MemorySearchResult, MemoryType


__all__ = [
    "BaseEmbedding",
    "BaseStore",
    "DistanceMetric",
    "MemoryRecord",
    "MemorySearchResult",
    "MemoryType",
    "OpenAIEmbedding",
]


try:
    from .qdrant_store import (
        QdrantStore,  # noqa: F401
        create_cloud_qdrant_store,  # noqa: F401
        create_local_qdrant_store,  # noqa: F401
        create_remote_qdrant_store,  # noqa: F401
    )

    __all__.extend(
        [
            "QdrantStore",
            "create_cloud_qdrant_store",
            "create_local_qdrant_store",
            "create_remote_qdrant_store",
        ]
    )
except ImportError:
    # qdrant-client not installed
    pass

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
