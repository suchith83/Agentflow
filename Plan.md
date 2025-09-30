PgCheckpointer (Persistent Layer)

Schema Evolution:
No versioning table or migration strategy beyond idempotent create.

Observability:
No metrics (timing, hit ratios, deserialization errors).

Error Handling:
Broad except Exception blocks without structured error categories (hard to automate retries precisely).

Background tasks manager exists but lacks cancellation propagation / structured lifecycle timeouts.

Potential event loop blocking via synchronous heavy JSON operations in tight loops (serialize each message individually).

DI invocation overhead per node (signature inspection once?—check if cached).

Logging at INFO in hot paths (state save, message store) may be chatty at scale—recommend structured DEBUG + counters instead.