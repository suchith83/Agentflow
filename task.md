# 📌 Improved Plan: Postgres + Redis Checkpointers for **PyAgenity**

## 1. Overview

We will create **Postgres** and **Redis** checkpointers extending from the `BaseCheckpointer` class.

* **Redis** → Used for fast state caching (`put_state_cache`, `get_state_cache`).
* **Postgres** → Used for persisting threads, states, and messages.
* **Async-first**: All operations use `asyncpg` and `aioredis`.
* **Thread lookups** are always done using `thread_id`.
* **User ID type** is configurable at initialization (`user_id_type="string"` by default).

---

## 2. Class Initialization

### PgCheckpointer

```python
class PgCheckpointer(BaseCheckpointer):
    def __init__(self, 
                 postgres_dsn: str, 
                 redis_url: str,
                 id_type: str = "string",        # thread_id, state_id, message_id type
                 user_id_type: str = "string"):  # explicitly set here, not via injectq
        self.postgres_dsn = postgres_dsn
        self.redis_url = redis_url
        self.id_type = id_type
        self.user_id_type = user_id_type
        self.pool = None
        self.redis = None
```

* `id_type` → `"string" | "int" | "bigint"`
* `user_id_type` → `"string" | "int" | "bigint"`, default `"string"`

---

## 3. Redis (Caching Layer)

* **Methods**:

  * `put_state_cache(config, state)` → Store state JSON with TTL.
  * `get_state_cache(config)` → Retrieve from Redis, fallback to Postgres if miss.
* **Key format**:

  ```
  state_cache:{thread_id}:{user_id}
  ```
* **Async client**: `aioredis.from_url(self.redis_url)`

---

## 4. Postgres (Persistence Layer)

### 4.1. ID Type Resolution

At setup, dynamically create schema depending on `id_type` and `user_id_type`.

| Config | SQL Type     |
| ------ | ------------ |
| string | VARCHAR(255) |
| int    | SERIAL       |
| bigint | BIGSERIAL    |

* If `generated_id` is provided → fallback to custom ID generator.
* `state_id` is optional, since lookups are always on `thread_id`.

---

### 4.2. Schema Design

#### 🔹 threads Table

```sql
CREATE TABLE IF NOT EXISTS threads (
    thread_id    {id_type} PRIMARY KEY,
    thread_name  VARCHAR(255),
    user_id      {user_id_type} NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    meta         JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_threads_user_id ON threads(user_id);
```

#### 🔹 states Table

```sql
CREATE TABLE IF NOT EXISTS states (
    state_id     serial PRIMARY KEY,
    thread_id    {id_type} NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    state_data   JSONB NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    meta         JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_states_thread_id ON states(thread_id);
```

#### 🔹 messages Table

```sql
CREATE TYPE IF NOT EXISTS message_role AS ENUM ('user', 'assistant', 'system', 'tool');

CREATE TABLE IF NOT EXISTS messages (
    message_id   {id_type} PRIMARY KEY,
    thread_id    {id_type} NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
    role         message_role NOT NULL,
    content      TEXT NOT NULL,
    tool_calls   JSONB,
    tool_call_id VARCHAR(255),
    reasoning    TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    total_tokens INT DEFAULT 0,
    usages       JSONB DEFAULT '{}'::jsonb,
    meta         JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);
```

---

## 6. Best Practices

* **Async pooling** for both Redis & Postgres.
* **SQL safety**: Always use parameterized queries.
* **Error handling**: Retry transient failures.
* **Consistency**: Redis → ephemeral, Postgres → source of truth.
* **Migrations**: Maintain `schema_migrations` table for version control.
* **Testing**: Cover schema creation, CRUD, Redis cache fallback, async race conditions.
* **Documentation**: Clear docstrings and usage examples.
* **Logging**: Use structured logging for operations and errors.
* **Resource management**: Ensure proper connection closing and cleanup.
* **Performance**: write efficient queries and use indexing, and it scalable, especially for high-throughput scenarios, consider connection pooling and caching strategies, and make sure all
async operations are non-blocking to maintain responsiveness, write efficient queries and use indexing.


### Starting code

```python
from typing import Literal

from asyncpg import Pool
from redis.asyncio import ConnectionPool, Redis

from .base_checkpointer import BaseCheckpointer


class PgCheckpointer(BaseCheckpointer):
    def __init__(
        self,
        # postgress connection details
        postgres_dsn: str | None = None,
        pg_pool: Pool | None = None,
        pool_config: dict | None = None,
        # redis connection details
        redis_url: str | None = None,
        redis: Redis | None = None,
        redis_pool: ConnectionPool | None = None,
        redis_pool_config: dict | None = None,
        # other configurations
        user_id_type: Literal["string", "int", "bigint"] = "string",
        release_resources: bool = False,
    ):
        self.user_id_type = user_id_type
        self.release_resources = release_resources

        # Now check and initialize connections
        if not pg_pool and not postgres_dsn:
            raise ValueError("Either postgres_dsn or pg_pool must be provided.")

        if not redis and not redis_url and not redis_pool:
            raise ValueError("Either redis_url, redis_pool or redis instance must be provided.")

        # Initialize connections
        self.pg_pool = self._create_pg_pool(pg_pool, postgres_dsn, pool_config or {})
        self.redis = self._create_redis_pool(redis_pool, redis_url, redis_pool_config or {})

    def _create_redis_pool(
        self,
        redis_pool,
        redis_url,
        redis_pool_config,
    ) -> Redis:
        if redis_pool:
            return Redis(connection_pool=redis_pool)

        # as we are creating new pool, redis_url must be provided
        # and we will release the resources if needed
        self.release_resources = True
        return Redis(
            connection_pool=ConnectionPool.from_url(
                redis_url,
                **redis_pool_config,
            )
        )

    def _create_pg_pool(self, pg_pool, postgres_dsn, pool_config) -> Pool:
        if pg_pool:
            return pg_pool
        import asyncpg  # noqa: PLC0415

        # as we are creating new pool, postgres_dsn must be provided
        # and we will release the resources if needed
        self.release_resources = True
        return asyncpg.create_pool(dsn=postgres_dsn, **pool_config)
``` 

Basic code structure is provided. The rest of the methods for CRUD operations, schema creation, and caching logic will follow this structure. Please do it step-by-step, ensuring each part is well-tested and documented.