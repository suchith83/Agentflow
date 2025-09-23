import asyncio
import json
import logging
from enum import Enum
from typing import Any, TypeVar

from injectq import InjectQ

from pyagenity.utils.thread_info import ThreadInfo


try:
    import asyncpg
    from asyncpg import Pool

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None  # type: ignore
    Pool = None  # type: ignore

try:
    from redis.asyncio import ConnectionPool, Redis

    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    ConnectionPool = None  # type: ignore
    Redis = None  # type: ignore

from pyagenity.state import AgentState
from pyagenity.utils import Message

from .base_checkpointer import BaseCheckpointer


logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound="AgentState")

# Default TTL for Redis cache (24 hours)
DEFAULT_CACHE_TTL = 86400

# SQL type mapping for ID types
ID_TYPE_MAP = {
    "string": "VARCHAR(255)",
    "int": "SERIAL",
    "bigint": "BIGSERIAL",
}


class PgCheckpointer(BaseCheckpointer[StateT]):
    """
    Postgres + Redis checkpointer implementation.

    Uses PostgreSQL for persistent storage of threads, states, and messages.
    Uses Redis for fast caching of state data with configurable TTL.

    Features:
    - Async-first design with sync fallbacks
    - Configurable ID types (string, int, bigint)
    - Connection pooling for both PostgreSQL and Redis
    - Proper error handling and resource management
    - Schema migration support

    Requires:
        pip install pyagenity[pg_checkpoint]
    """

    def __init__(
        self,
        # postgress connection details
        postgres_dsn: str | None = None,
        pg_pool: Any | None = None,
        pool_config: dict | None = None,
        # redis connection details
        redis_url: str | None = None,
        redis: Any | None = None,
        redis_pool: Any | None = None,
        redis_pool_config: dict | None = None,
        # other configurations - combine to reduce args
        **kwargs,
    ):
        """
        Initialize PgCheckpointer with PostgreSQL and Redis connections.

        Args:
            postgres_dsn: PostgreSQL connection string
            pg_pool: Existing asyncpg Pool instance
            pool_config: Configuration for new pg pool creation
            redis_url: Redis connection URL
            redis: Existing Redis instance
            redis_pool: Existing Redis ConnectionPool
            redis_pool_config: Configuration for new redis pool creation
            **kwargs: Additional configuration options:
                - user_id_type: Type for user_id fields ('string', 'int', 'bigint')
                - cache_ttl: Redis cache TTL in seconds
                - release_resources: Whether to release resources on cleanup
        """
        # Check for required dependencies
        if not HAS_ASYNCPG:
            raise ImportError(
                "PgCheckpointer requires 'asyncpg' package. "
                "Install with: pip install pyagenity[pg_checkpoint]"
            )

        if not HAS_REDIS:
            raise ImportError(
                "PgCheckpointer requires 'redis' package. "
                "Install with: pip install pyagenity[pg_checkpoint]"
            )

        self.user_id_type = kwargs.get("user_id_type", "string")
        self.id_type = InjectQ.get_instance().try_get("generated_id_type", "string")
        self.cache_ttl = kwargs.get("cache_ttl", DEFAULT_CACHE_TTL)
        self.release_resources = kwargs.get("release_resources", False)
        self._schema_initialized = False

        # Store pool configuration for lazy initialization
        self._pg_pool_config = {
            "pg_pool": pg_pool,
            "postgres_dsn": postgres_dsn,
            "pool_config": pool_config or {},
        }

        # Initialize pool immediately if provided, otherwise defer
        if pg_pool is not None:
            self._pg_pool = pg_pool
        else:
            self._pg_pool = None

        # Now check and initialize connections
        if not pg_pool and not postgres_dsn:
            raise ValueError("Either postgres_dsn or pg_pool must be provided.")

        if not redis and not redis_url and not redis_pool:
            raise ValueError("Either redis_url, redis_pool or redis instance must be provided.")

        # Initialize Redis connection (synchronous)
        self.redis = self._create_redis_pool(redis, redis_pool, redis_url, redis_pool_config or {})

    def _create_redis_pool(
        self,
        redis: Any | None,
        redis_pool: Any | None,
        redis_url: str | None,
        redis_pool_config: dict,
    ) -> Any:
        """Create or use existing Redis connection."""
        if redis:
            return redis

        if redis_pool:
            return Redis(connection_pool=redis_pool)  # type: ignore

        # as we are creating new pool, redis_url must be provided
        # and we will release the resources if needed
        if not redis_url:
            raise ValueError("redis_url must be provided when creating new Redis connection")

        self.release_resources = True
        return Redis(  # type: ignore
            connection_pool=ConnectionPool.from_url(  # type: ignore
                redis_url,
                **redis_pool_config,
            )
        )

    def _create_pg_pool(self, pg_pool: Any, postgres_dsn: str | None, pool_config: dict) -> Any:
        if pg_pool:
            return pg_pool
        # as we are creating new pool, postgres_dsn must be provided
        # and we will release the resources if needed
        self.release_resources = True
        return asyncpg.create_pool(dsn=postgres_dsn, **pool_config)  # type: ignore

    async def _get_pg_pool(self) -> Any:
        """Get PostgreSQL pool, creating it if necessary."""
        if self._pg_pool is None:
            config = self._pg_pool_config
            self._pg_pool = self._create_pg_pool(
                config["pg_pool"], config["postgres_dsn"], config["pool_config"]
            )
        return self._pg_pool

    def _get_sql_type(self, type_name: str) -> str:
        """Get SQL type for given configuration type."""
        return ID_TYPE_MAP.get(type_name, "VARCHAR(255)")

    def _build_create_tables_sql(self) -> list[str]:
        """Build SQL statements for table creation with dynamic ID types."""
        thread_id_type = self._get_sql_type(self.id_type)
        user_id_type = self._get_sql_type(self.user_id_type)
        message_id_type = self._get_sql_type(self.id_type)

        # For AUTO INCREMENT types, we need to handle primary key differently
        thread_pk = (
            "thread_id SERIAL PRIMARY KEY"
            if self.id_type == "int"
            else f"thread_id {thread_id_type} PRIMARY KEY"
        )
        message_pk = (
            "message_id SERIAL PRIMARY KEY"
            if self.id_type == "int"
            else f"message_id {message_id_type} PRIMARY KEY"
        )

        return [
            # Create message role enum
            (
                "CREATE TYPE IF NOT EXISTS message_role AS ENUM "
                "('user', 'assistant', 'system', 'tool')"
            ),
            # Create threads table
            f"""
            CREATE TABLE IF NOT EXISTS threads (
                {thread_pk},
                thread_name VARCHAR(255),
                user_id {user_id_type} NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                meta JSONB DEFAULT '{{}}'::jsonb
            )
            """,
            # Create states table
            f"""
            CREATE TABLE IF NOT EXISTS states (
                state_id SERIAL PRIMARY KEY,
                thread_id {thread_id_type} NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
                state_data JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                meta JSONB DEFAULT '{{}}'::jsonb
            )
            """,
            # Create messages table
            f"""
            CREATE TABLE IF NOT EXISTS messages (
                {message_pk},
                thread_id {thread_id_type} NOT NULL REFERENCES threads(thread_id) ON DELETE CASCADE,
                role message_role NOT NULL,
                content TEXT NOT NULL,
                tool_calls JSONB,
                tool_call_id VARCHAR(255),
                reasoning TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                total_tokens INT DEFAULT 0,
                usages JSONB DEFAULT '{{}}'::jsonb,
                meta JSONB DEFAULT '{{}}'::jsonb
            )
            """,
            # Create indexes
            "CREATE INDEX IF NOT EXISTS idx_threads_user_id ON threads(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_states_thread_id ON states(thread_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id)",
        ]

    async def _initialize_schema(self) -> None:
        """Initialize database schema if not already done."""
        if self._schema_initialized:
            return

        logger.info(
            "Initializing database schema with types: id_type=%s, user_id_type=%s",
            self.id_type,
            self.user_id_type,
        )

        async with (await self._get_pg_pool()).acquire() as conn:
            try:
                sql_statements = self._build_create_tables_sql()
                for sql in sql_statements:
                    logger.debug("Executing SQL: %s", sql.strip())
                    await conn.execute(sql)
                self._schema_initialized = True
                logger.info("Database schema initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize database schema: %s", e)
                raise

    ###########################
    #### SETUP METHODS ########
    ###########################

    def setup(self) -> Any:
        """Sync setup method - runs schema initialization."""
        logger.info("Setting up PgCheckpointer (sync)")
        return asyncio.run(self.asetup())

    async def asetup(self) -> Any:
        """Async setup method - initializes database schema."""
        logger.info("Setting up PgCheckpointer (async)")
        await self._initialize_schema()
        logger.info("PgCheckpointer setup completed")
        return True

    ###########################
    #### HELPER METHODS #######
    ###########################

    def _validate_config(self, config: dict[str, Any]) -> tuple[str | int, str | int]:
        """Extract and validate thread_id and user_id from config."""
        thread_id = config.get("thread_id")
        user_id = config.get("user_id")

        if not thread_id or not user_id:
            raise ValueError("Both thread_id and user_id must be provided in config")

        return thread_id, user_id

    def _get_thread_key(
        self,
        thread_id: str | int,
        user_id: str | int,
    ) -> str:
        """Get Redis cache key for thread state."""
        return f"state_cache:{thread_id}:{user_id}"

    def _serialize_state(self, state: StateT) -> str:
        """Serialize state to JSON string for storage."""

        def enum_handler(obj):
            if isinstance(obj, Enum):
                return obj.value
            return str(obj)

        return json.dumps(state.model_dump(), default=enum_handler)

    def _deserialize_state(
        self,
        data: str,
        state_class: type[StateT],
    ) -> StateT:
        """Deserialize JSON string back to state object."""
        return state_class.model_validate(json.loads(data))

    async def _run_sync(
        self,
        func,
        *args,
        **kwargs,
    ):
        """Run a synchronous function in a thread pool."""
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _retry_on_connection_error(
        self,
        operation,
        *args,
        max_retries=3,
        **kwargs,
    ):
        """Retry database operations on connection errors."""
        last_exception = None

        # Define exception types to catch (only if asyncpg is available)
        exceptions_to_catch: list[type[Exception]] = [ConnectionError]
        if HAS_ASYNCPG and asyncpg:
            exceptions_to_catch.extend([asyncpg.PostgresConnectionError, asyncpg.InterfaceError])

        exception_tuple = tuple(exceptions_to_catch)

        for attempt in range(max_retries):
            try:
                return await operation(*args, **kwargs)
            except exception_tuple as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # exponential backoff
                    logger.warning(
                        "Database connection error on attempt %d/%d, retrying in %ds: %s",
                        attempt + 1,
                        max_retries,
                        wait_time,
                        e,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                logger.error("Failed after %d attempts: %s", max_retries, e)
                break
            except Exception as e:
                # Don't retry on non-connection errors
                logger.error("Non-retryable error: %s", e)
                raise

        if last_exception:
            raise last_exception
        return None

    ###########################
    #### STATE METHODS ########
    ###########################

    async def aput_state(
        self,
        config: dict[str, Any],
        state: StateT,
    ) -> StateT:
        """Store state in PostgreSQL and optionally cache in Redis."""
        thread_id, user_id = self._validate_config(config)

        logger.debug("Storing state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            # Ensure thread exists first
            await self._ensure_thread_exists(thread_id, user_id, config)

            # Store in PostgreSQL with retry logic
            state_json = self._serialize_state(state)

            async def _store_state():
                async with (await self._get_pg_pool()).acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO states (thread_id, state_data, meta)
                        VALUES ($1, $2, $3)
                        ON CONFLICT DO NOTHING
                        """,
                        thread_id,
                        state_json,
                        config.get("meta", {}),
                    )

            await self._retry_on_connection_error(_store_state, max_retries=3)
            logger.info("State stored successfully for thread_id=%s", thread_id)
            return state

        except Exception as e:
            logger.error("Failed to store state for thread_id=%s: %s", thread_id, e)
            raise

    async def aget_state(self, config: dict[str, Any]) -> StateT | None:
        """Retrieve state from PostgreSQL."""
        thread_id, user_id = self._validate_config(config)
        state_class = config.get("state_class", AgentState)

        logger.debug("Retrieving state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:

            async def _get_state():
                async with (await self._get_pg_pool()).acquire() as conn:
                    return await conn.fetchrow(
                        """
                        SELECT state_data FROM states
                        WHERE thread_id = $1
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        thread_id,
                    )

            row = await self._retry_on_connection_error(_get_state, max_retries=3)

            if row:
                logger.debug("State found for thread_id=%s", thread_id)
                return self._deserialize_state(row["state_data"], state_class)

            logger.debug("No state found for thread_id=%s", thread_id)
            return None

        except Exception as e:
            logger.error("Failed to retrieve state for thread_id=%s: %s", thread_id, e)
            raise

    async def aclear_state(self, config: dict[str, Any]) -> Any:
        """Clear state from PostgreSQL and Redis cache."""
        thread_id, user_id = self._validate_config(config)

        logger.debug("Clearing state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            # Clear from PostgreSQL with retry logic
            async def _clear_state():
                async with (await self._get_pg_pool()).acquire() as conn:
                    await conn.execute("DELETE FROM states WHERE thread_id = $1", thread_id)

            await self._retry_on_connection_error(_clear_state, max_retries=3)

            # Clear from Redis cache
            cache_key = self._get_thread_key(thread_id, user_id)
            await self.redis.delete(cache_key)

            logger.info("State cleared for thread_id=%s", thread_id)

        except Exception as e:
            logger.error("Failed to clear state for thread_id=%s: %s", thread_id, e)
            raise

    async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        """Cache state in Redis with TTL."""
        thread_id, user_id = self._validate_config(config)

        logger.debug("Caching state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            cache_key = self._get_thread_key(thread_id, user_id)
            state_json = self._serialize_state(state)
            await self.redis.setex(cache_key, self.cache_ttl, state_json)
            logger.debug("State cached with key=%s, ttl=%d", cache_key, self.cache_ttl)
            return True

        except Exception as e:
            logger.error("Failed to cache state for thread_id=%s: %s", thread_id, e)
            # Don't raise - caching is optional
            return None

    async def aget_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """Get state from Redis cache, fallback to PostgreSQL if miss."""
        thread_id, user_id = self._validate_config(config)
        state_class = config.get("state_class", AgentState)

        logger.debug("Getting cached state for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            # Try Redis first
            cache_key = self._get_thread_key(thread_id, user_id)
            cached_data = await self.redis.get(cache_key)

            if cached_data:
                logger.debug("Cache hit for thread_id=%s", thread_id)
                return self._deserialize_state(cached_data.decode(), state_class)

            # Cache miss - fallback to PostgreSQL
            logger.debug("Cache miss for thread_id=%s, falling back to PostgreSQL", thread_id)
            state = await self.aget_state(config)

            # Cache the result for next time
            if state:
                await self.aput_state_cache(config, state)

            return state

        except Exception as e:
            logger.error("Failed to get cached state for thread_id=%s: %s", thread_id, e)
            # Fallback to PostgreSQL on error
            return await self.aget_state(config)

    async def _ensure_thread_exists(
        self,
        thread_id: str | int,
        user_id: str | int,
        config: dict[str, Any],
    ) -> None:
        """Ensure thread exists in database, create if not."""
        try:

            async def _check_and_create_thread():
                async with (await self._get_pg_pool()).acquire() as conn:
                    exists = await conn.fetchval(
                        "SELECT 1 FROM threads WHERE thread_id = $1 AND user_id = $2",
                        thread_id,
                        user_id,
                    )

                    if not exists:
                        thread_name = config.get("thread_name", f"Thread {thread_id}")
                        meta = config.get("thread_meta", {})
                        await conn.execute(
                            """
                            INSERT INTO threads (thread_id, thread_name, user_id, meta)
                            VALUES ($1, $2, $3, $4)
                            ON CONFLICT DO NOTHING
                            """,
                            thread_id,
                            thread_name,
                            user_id,
                            meta,
                        )
                        logger.debug("Created thread: thread_id=%s, user_id=%s", thread_id, user_id)

            await self._retry_on_connection_error(_check_and_create_thread, max_retries=3)

        except Exception as e:
            logger.error("Failed to ensure thread exists: %s", e)
            raise

    # Sync variants
    def put_state(self, config: dict[str, Any], state: StateT) -> StateT:
        """Sync version of put_state."""
        return asyncio.run(self.aput_state(config, state))

    def get_state(self, config: dict[str, Any]) -> StateT | None:
        """Sync version of get_state."""
        return asyncio.run(self.aget_state(config))

    def clear_state(self, config: dict[str, Any]) -> Any:
        """Sync version of clear_state."""
        return asyncio.run(self.aclear_state(config))

    def put_state_cache(self, config: dict[str, Any], state: StateT) -> Any | None:
        """Sync version of put_state_cache."""
        return asyncio.run(self.aput_state_cache(config, state))

    def get_state_cache(self, config: dict[str, Any]) -> StateT | None:
        """Sync version of get_state_cache."""
        return asyncio.run(self.aget_state_cache(config))

    ###########################
    #### MESSAGE METHODS ######
    ###########################

    async def aput_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Store messages in PostgreSQL."""
        thread_id, user_id = self._validate_config(config)

        if not messages:
            logger.debug("No messages to store for thread_id=%s", thread_id)
            return

        logger.debug("Storing %d messages for thread_id=%s", len(messages), thread_id)

        try:
            # Ensure thread exists
            await self._ensure_thread_exists(thread_id, user_id, config)

            # Store messages in batch with retry logic
            async def _store_messages():
                async with (await self._get_pg_pool()).acquire() as conn, conn.transaction():
                    for message in messages:
                        await conn.execute(
                            """
                                INSERT INTO messages (
                                    message_id, thread_id, role, content, tool_calls,
                                    tool_call_id, reasoning, total_tokens, usages, meta
                                )
                                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                                ON CONFLICT (message_id) DO UPDATE SET
                                    content = EXCLUDED.content,
                                    reasoning = EXCLUDED.reasoning,
                                    usages = EXCLUDED.usages,
                                    updated_at = NOW()
                                """,
                            message.message_id,
                            thread_id,
                            message.role,
                            message.content,
                            json.dumps(message.tools_calls) if message.tools_calls else None,
                            message.reasoning,
                            message.usages.total_tokens if message.usages else 0,
                            json.dumps(message.usages.model_dump()) if message.usages else None,
                            json.dumps({**(metadata or {}), **message.metadata}),
                        )

            await self._retry_on_connection_error(_store_messages, max_retries=3)
            logger.info("Stored %d messages for thread_id=%s", len(messages), thread_id)

        except Exception as e:
            logger.error("Failed to store messages for thread_id=%s: %s", thread_id, e)
            raise

    async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
        """Retrieve a single message by ID."""
        thread_id = config.get("thread_id")

        logger.debug("Retrieving message_id=%s for thread_id=%s", message_id, thread_id)

        try:

            async def _get_message():
                async with (await self._get_pg_pool()).acquire() as conn:
                    query = """
                        SELECT message_id, thread_id, role, content, tool_calls,
                               tool_call_id, reasoning, created_at, total_tokens,
                               usages, meta
                        FROM messages
                        WHERE message_id = $1
                    """
                    if thread_id:
                        query += " AND thread_id = $2"
                        return await conn.fetchrow(query, message_id, thread_id)
                    return await conn.fetchrow(query, message_id)

            row = await self._retry_on_connection_error(_get_message, max_retries=3)

            if not row:
                raise ValueError(f"Message not found: {message_id}")

            return self._row_to_message(row)

        except Exception as e:
            logger.error("Failed to retrieve message_id=%s: %s", message_id, e)
            raise

    async def alist_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """List messages for a thread with optional search and pagination."""
        thread_id = config.get("thread_id")

        if not thread_id:
            raise ValueError("thread_id must be provided in config")

        logger.debug("Listing messages for thread_id=%s", thread_id)

        try:

            async def _list_messages():
                async with (await self._get_pg_pool()).acquire() as conn:
                    # Build query with optional search
                    query = """
                        SELECT message_id, thread_id, role, content, tool_calls,
                               tool_call_id, reasoning, created_at, total_tokens,
                               usages, meta
                        FROM messages
                        WHERE thread_id = $1
                    """
                    params = [thread_id]
                    param_count = 1

                    if search:
                        param_count += 1
                        query += f" AND content ILIKE ${param_count}"
                        params.append(f"%{search}%")

                    query += " ORDER BY created_at ASC"

                    if limit:
                        param_count += 1
                        query += f" LIMIT ${param_count}"
                        params.append(limit)

                    if offset:
                        param_count += 1
                        query += f" OFFSET ${param_count}"
                        params.append(offset)

                    return await conn.fetch(query, *params)

            rows = await self._retry_on_connection_error(_list_messages, max_retries=3)
            if not rows:
                rows = []
            messages = [self._row_to_message(row) for row in rows]

            logger.debug("Found %d messages for thread_id=%s", len(messages), thread_id)
            return messages

        except Exception as e:
            logger.error("Failed to list messages for thread_id=%s: %s", thread_id, e)
            raise

    async def adelete_message(
        self,
        config: dict[str, Any],
        message_id: str | int,
    ) -> Any | None:
        """Delete a message by ID."""
        thread_id = config.get("thread_id")

        logger.debug("Deleting message_id=%s for thread_id=%s", message_id, thread_id)

        try:

            async def _delete_message():
                async with (await self._get_pg_pool()).acquire() as conn:
                    if thread_id:
                        await conn.execute(
                            "DELETE FROM messages WHERE message_id = $1 AND thread_id = $2",
                            message_id,
                            thread_id,
                        )
                    else:
                        await conn.execute("DELETE FROM messages WHERE message_id = $1", message_id)

            await self._retry_on_connection_error(_delete_message, max_retries=3)
            logger.info("Deleted message_id=%s", message_id)
            return None

        except Exception as e:
            logger.error("Failed to delete message_id=%s: %s", message_id, e)
            raise

    def _row_to_message(self, row) -> Message:
        """Convert database row to Message object."""
        from pyagenity.utils.message import TokenUsages

        usages = None
        if row["usages"]:
            usages_data = json.loads(row["usages"])
            usages = TokenUsages(**usages_data)

        return Message(
            message_id=row["message_id"],
            role=row["role"],
            content=row["content"],
            tools_calls=json.loads(row["tool_calls"]) if row["tool_calls"] else None,
            reasoning=row["reasoning"],
            timestamp=row["created_at"],
            metadata=json.loads(row["meta"]) if row["meta"] else {},
            usages=usages,
        )

    # Sync variants
    def put_messages(
        self,
        config: dict[str, Any],
        messages: list[Message],
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Sync version of put_messages."""
        return asyncio.run(self.aput_messages(config, messages, metadata))

    def get_message(self, config: dict[str, Any]) -> Message:
        """Sync version of get_message."""
        message_id = config.get("message_id")
        if not message_id:
            raise ValueError("message_id must be provided in config")
        return asyncio.run(self.aget_message(config, message_id))

    def list_messages(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[Message]:
        """Sync version of list_messages."""
        return asyncio.run(self.alist_messages(config, search, offset, limit))

    def delete_message(self, config: dict[str, Any], message_id: str | int) -> Any | None:
        """Sync version of delete_message."""
        return asyncio.run(self.adelete_message(config, message_id))

    ###########################
    #### THREAD METHODS #######
    ###########################

    async def aput_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> Any | None:
        """Create or update thread information."""
        thread_id, user_id = self._validate_config(config)

        logger.debug("Storing thread info for thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            thread_name = thread_info.thread_name or f"Thread {thread_id}"
            meta = thread_info.metadata or {}
            user_id = thread_info.user_id or user_id

            async def _put_thread():
                async with (await self._get_pg_pool()).acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO threads (thread_id, thread_name, user_id, meta)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (thread_id) DO UPDATE SET
                            thread_name = EXCLUDED.thread_name,
                            meta = EXCLUDED.meta,
                            updated_at = NOW()
                        """,
                        thread_id,
                        thread_name,
                        user_id,
                        meta,
                    )

            await self._retry_on_connection_error(_put_thread, max_retries=3)
            logger.info("Thread info stored for thread_id=%s", thread_id)

        except Exception as e:
            logger.error("Failed to store thread info for thread_id=%s: %s", thread_id, e)
            raise

    async def aget_thread(
        self,
        config: dict[str, Any],
    ) -> ThreadInfo | None:
        """Get thread information."""
        thread_id, user_id = self._validate_config(config)

        logger.debug("Retrieving thread info for thread_id=%s, user_id=%s", thread_id, user_id)

        try:

            async def _get_thread():
                async with (await self._get_pg_pool()).acquire() as conn:
                    return await conn.fetchrow(
                        """
                        SELECT thread_id, thread_name, user_id, created_at, updated_at, meta
                        FROM threads
                        WHERE thread_id = $1 AND user_id = $2
                        """,
                        thread_id,
                        user_id,
                    )

            row = await self._retry_on_connection_error(_get_thread, max_retries=3)

            if row:
                return ThreadInfo(
                    thread_id=thread_id,
                    thread_name=row["thread_name"] if row else None,
                    user_id=user_id,
                    metadata=row["meta"] if row else {},
                )

            logger.debug("Thread not found for thread_id=%s, user_id=%s", thread_id, user_id)
            return None

        except Exception as e:
            logger.error("Failed to retrieve thread info for thread_id=%s: %s", thread_id, e)
            raise

    async def alist_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """List threads for a user with optional search and pagination."""
        user_id = config.get("user_id")

        if not user_id:
            raise ValueError("user_id must be provided in config")

        logger.debug("Listing threads for user_id=%s", user_id)

        try:

            async def _list_threads():
                async with (await self._get_pg_pool()).acquire() as conn:
                    # Build query with optional search
                    query = """
                        SELECT thread_id, thread_name, user_id, created_at, updated_at, meta
                        FROM threads
                        WHERE user_id = $1
                    """
                    params = [user_id]
                    param_count = 1

                    if search:
                        param_count += 1
                        query += f" AND thread_name ILIKE ${param_count}"
                        params.append(f"%{search}%")

                    query += " ORDER BY updated_at DESC"

                    if limit:
                        param_count += 1
                        query += f" LIMIT ${param_count}"
                        params.append(limit)

                    if offset:
                        param_count += 1
                        query += f" OFFSET ${param_count}"
                        params.append(offset)

                    return await conn.fetch(query, *params)

            rows = await self._retry_on_connection_error(_list_threads, max_retries=3)
            if not rows:
                rows = []

            threads = [
                ThreadInfo(
                    thread_id=row["thread_id"],
                    thread_name=row["thread_name"],
                    user_id=row["user_id"],
                    metadata=row["meta"] or {},
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

            logger.debug("Found %d threads for user_id=%s", len(threads), user_id)
            return threads

        except Exception as e:
            logger.error("Failed to list threads for user_id=%s: %s", user_id, e)
            raise

    async def aclean_thread(self, config: dict[str, Any]) -> Any | None:
        """Clean/delete a thread and all associated data."""
        thread_id, user_id = self._validate_config(config)

        logger.debug("Cleaning thread thread_id=%s, user_id=%s", thread_id, user_id)

        try:
            # Delete thread (cascade will handle messages and states) with retry logic
            async def _clean_thread():
                async with (await self._get_pg_pool()).acquire() as conn:
                    await conn.execute(
                        "DELETE FROM threads WHERE thread_id = $1 AND user_id = $2",
                        thread_id,
                        user_id,
                    )

            await self._retry_on_connection_error(_clean_thread, max_retries=3)

            # Clean from Redis cache
            cache_key = self._get_thread_key(thread_id, user_id)
            await self.redis.delete(cache_key)

            logger.info("Thread cleaned: thread_id=%s, user_id=%s", thread_id, user_id)

        except Exception as e:
            logger.error("Failed to clean thread thread_id=%s: %s", thread_id, e)
            raise

    # Sync variants
    def put_thread(
        self,
        config: dict[str, Any],
        thread_info: ThreadInfo,
    ) -> Any | None:
        """Sync version of put_thread."""
        return asyncio.run(self.aput_thread(config, thread_info))

    def get_thread(self, config: dict[str, Any]) -> ThreadInfo | None:
        """Sync version of get_thread."""
        return asyncio.run(self.aget_thread(config))

    def list_threads(
        self,
        config: dict[str, Any],
        search: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> list[ThreadInfo]:
        """Sync version of list_threads."""
        return asyncio.run(self.alist_threads(config, search, offset, limit))

    def clean_thread(self, config: dict[str, Any]) -> Any | None:
        """Sync version of clean_thread."""
        return asyncio.run(self.aclean_thread(config))

    ###########################
    #### RESOURCE CLEANUP #####
    ###########################

    def release(self) -> Any | None:
        """Sync version of resource cleanup."""
        return asyncio.run(self.arelease())

    async def arelease(self) -> Any | None:
        """Clean up connections and resources."""
        logger.info("Releasing PgCheckpointer resources")

        if not self.release_resources:
            logger.info("No resources to release")
            return

        errors = []

        # Close Redis connection
        try:
            if hasattr(self.redis, "aclose"):
                await self.redis.aclose()
            elif hasattr(self.redis, "close"):
                await self.redis.close()
            logger.debug("Redis connection closed")
        except Exception as e:
            logger.error("Error closing Redis connection: %s", e)
            errors.append(f"Redis: {e}")

        # Close PostgreSQL pool
        try:
            if self._pg_pool and not self._pg_pool.is_closing():
                await self._pg_pool.close()
            logger.debug("PostgreSQL pool closed")
        except Exception as e:
            logger.error("Error closing PostgreSQL pool: %s", e)
            errors.append(f"PostgreSQL: {e}")

        if errors:
            error_msg = f"Errors during resource cleanup: {'; '.join(errors)}"
            logger.warning(error_msg)
            # Don't raise - cleanup should be best effort
        else:
            logger.info("All resources released successfully")
