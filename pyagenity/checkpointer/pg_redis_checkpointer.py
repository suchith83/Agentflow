# import asyncio
# import logging
# from typing import Any, Optional, TypeVar

# import aioredis
# import asyncpg

# from pyagenity.checkpointer.base_checkpointer import BaseCheckpointer
# from pyagenity.state import AgentState
# from pyagenity.utils import Message


# logger = logging.getLogger(__name__)


# class PgRedisCheckpointer(BaseCheckpointer[AgentState]):
#     """
#     Checkpointer implementation using asyncpg for PostgreSQL and aioredis for Redis cache.
#     Manages its own connection pools and provides both async and sync methods.
#     Optional: save_raw controls whether to persist the raw field in messages.
#     """

#     def __init__(self, pg_dsn: str, redis_url: str, save_raw: bool = False):
#         self.pg_dsn = pg_dsn
#         self.redis_url = redis_url
#         self.save_raw = save_raw
#         self.pg_pool: Optional[asyncpg.Pool] = None
#         self.redis: Optional[aioredis.Redis] = None

#         # Runtime check for asyncpg and aioredis
#         self._asyncpg_available = False
#         self._aioredis_available = False
#         try:
#             import asyncpg  # noqa: F401

#             self._asyncpg_available = True
#         except ImportError:
#             logger.warning("asyncpg not available, DB operations will be disabled.")
#         try:
#             import aioredis  # noqa: F401

#             self._aioredis_available = True
#         except ImportError:
#             logger.warning("aioredis not available, cache operations will be disabled.")

#     async def asetup(self, config: dict[str, Any]) -> None:
#         """Initialize asyncpg and redis pools, and create tables if not exist."""
#         self.pg_pool = await asyncpg.create_pool(dsn=self.pg_dsn)
#         self.redis = await aioredis.from_url(self.redis_url, decode_responses=False)
#         logger.info("PgRedisCheckpointer pools initialized.")

#         # Create tables if not exist
#         create_thread = """
#         CREATE TABLE IF NOT EXISTS thread (
#             thread_id INTEGER PRIMARY KEY,
#             user_id INTEGER NULL,
#             thread_name TEXT NOT NULL,
#             meta JSONB NULL,
#             created_at TIMESTAMP DEFAULT NOW(),
#             updated_at TIMESTAMP DEFAULT NOW()
#         );
#         """
#         # TODO: Change Message ID
#         create_message_table = """
#         CREATE TABLE IF NOT EXISTS message_table (
#             message_id VARCHAR(64) PRIMARY KEY,
#             user_id INTEGER NULL,
#             thread_id INTEGER REFERENCES thread(thread_id) ON DELETE CASCADE,
#             role VARCHAR(16) NOT NULL,
#             content TEXT NOT NULL,
#             tools_calls JSONB,
#             tool_call_id VARCHAR(64),
#             function_call JSONB,
#             reasoning TEXT,
#             timestamp TIMESTAMP,
#             metadata JSONB,
#             usages JSONB,
#             raw JSONB,
#             created_at TIMESTAMP DEFAULT NOW(),
#             updated_at TIMESTAMP DEFAULT NOW()
#         );
#         """
#         create_state = """
#         CREATE TABLE IF NOT EXISTS state (
#             state_id SERIAL PRIMARY KEY,
#             user_id INTEGER NULL,
#             thread_id INTEGER REFERENCES thread(thread_id) ON DELETE CASCADE,
#             state_data JSONB NOT NULL,
#             created_at TIMESTAMP DEFAULT NOW(),
#             updated_at TIMESTAMP DEFAULT NOW()
#         );
#         """
#         async with self.pg_pool.acquire() as conn:
#             await conn.execute(create_thread)
#             await conn.execute(create_message_table)
#             await conn.execute(create_state)
#         logger.info("PgRedisCheckpointer tables ensured.")

#     def setup(self, config: dict[str, Any]) -> None:
#         """Sync setup using asyncio.run."""
#         asyncio.run(self.asetup(config))

#     # -------------------------
#     # State methods Async
#     # -------------------------
#     async def aput_state(self, config: dict[str, Any], state: StateT) -> StateT:
#         """Persist state in the database and update cache if available."""
#         if not self._asyncpg_available or not self.pg_pool:
#             logger.error("asyncpg is not available or not initialized.")
#             return state
#         user_id = config.get("user")
#         thread_id = config.get("thread_id")
#         async with self.pg_pool.acquire() as conn:
#             await conn.execute(
#                 """
#                 INSERT INTO state (user_id, thread_id, state_data, created_at, updated_at)
#                 VALUES ($1, $2, $3, NOW(), NOW())
#                 ON CONFLICT (thread_id) DO UPDATE
#                 SET state_data = $3, updated_at = NOW()
#                 """,
#                 user_id,
#                 thread_id,
#                 state.model_dump_json(),
#             )
#         # Cache in redis if available
#         if self._aioredis_available and self.redis:
#             await self.redis.set(f"state:{thread_id}", state.model_dump_json())
#         return state

#     async def aget_state(self, config: dict[str, Any]) -> Optional[StateT]:
#         """Retrieve state from cache or database."""
#         thread_id = config.get("thread_id")
#         user_id = config.get("user")
#         # Try cache first
#         if self._aioredis_available and self.redis:
#             cached = await self.redis.get(f"state:{thread_id}")
#             if cached:
#                 return AgentState.model_validate_json(cached)
#         # Fallback to DB
#         if not self._asyncpg_available or not self.pg_pool:
#             logger.error("asyncpg is not available or not initialized.")
#             return None
#         async with self.pg_pool.acquire() as conn:
#             row = await conn.fetchrow(
#                 "SELECT state_data FROM state WHERE thread_id = $1 AND (user_id = $2 OR $2 IS NULL)",
#                 thread_id,
#                 user_id,
#             )
#             if row:
#                 return AgentState.model_validate_json(row["state_data"])
#         return None

#     async def aclear_state(self, config: dict[str, Any]) -> Any:
#         """Delete state from DB and cache."""
#         thread_id = config.get("thread_id")
#         user_id = config.get("user")
#         if self._asyncpg_available and self.pg_pool:
#             async with self.pg_pool.acquire() as conn:
#                 await conn.execute(
#                     "DELETE FROM state WHERE thread_id = $1 AND (user_id = $2 OR $2 IS NULL)",
#                     thread_id,
#                     user_id,
#                 )
#         if self._aioredis_available and self.redis:
#             await self.redis.delete(f"state:{thread_id}")
#         return True

#     async def aput_state_cache(self, config: dict[str, Any], state: StateT) -> Any:
#         """Put state in cache only."""
#         thread_id = config.get("thread_id")
#         if self._aioredis_available and self.redis:
#             await self.redis.set(f"state:{thread_id}", state.model_dump_json())
#             return True
#         return False

#     async def aget_state_cache(self, config: dict[str, Any]) -> Optional[StateT]:
#         """Get state from cache only."""
#         thread_id = config.get("thread_id")
#         if self._aioredis_available and self.redis:
#             cached = await self.redis.get(f"state:{thread_id}")
#             if cached:
#                 return AgentState.model_validate_json(cached)
#         return None

#     # -------------------------
#     # State methods Sync
#     # -------------------------
#     def put_state(self, config: dict[str, Any], state: StateT) -> StateT:
#         return asyncio.run(self.aput_state(config, state))

#     def get_state(self, config: dict[str, Any]) -> Optional[StateT]:
#         return asyncio.run(self.aget_state(config))

#     def clear_state(self, config: dict[str, Any]) -> Any:
#         return asyncio.run(self.aclear_state(config))

#     def put_state_cache(self, config: dict[str, Any], state: StateT) -> Any:
#         return asyncio.run(self.aput_state_cache(config, state))

#     def get_state_cache(self, config: dict[str, Any]) -> Optional[StateT]:
#         return asyncio.run(self.aget_state_cache(config))

#     # -------------------------
#     # Message methods async
#     # -------------------------
#     async def aput_messages(
#         self,
#         config: dict[str, Any],
#         messages: list[Message],
#         metadata: dict[str, Any] | None = None,
#     ) -> Any:
#         """Persist messages in the database, respecting save_raw."""
#         if not self._asyncpg_available or not self.pg_pool:
#             logger.error("asyncpg is not available or not initialized.")
#             return False
#         user_id = config.get("user")
#         thread_id = config.get("thread_id")
#         async with self.pg_pool.acquire() as conn:
#             for msg in messages:
#                 await conn.execute(
#                     """
#                     INSERT INTO message_table (
#                         message_id, user_id, thread_id, role, content, tools_calls, tool_call_id,
#                         function_call, reasoning, timestamp, metadata, usages, raw, created_at, updated_at
#                     ) VALUES (
#                         $1, $2, $3, $4, $5, $6, $7,
#                         $8, $9, $10, $11, $12, $13, NOW(), NOW()
#                     )
#                     ON CONFLICT (message_id) DO UPDATE
#                     SET content = $5, tools_calls = $6, tool_call_id = $7,
#                         function_call = $8, reasoning = $9, timestamp = $10,
#                         metadata = $11, usages = $12, raw = $13, updated_at = NOW()
#                     """,
#                     msg.message_id,
#                     user_id,
#                     thread_id,
#                     msg.role,
#                     msg.content,
#                     getattr(msg, "tools_calls", None),
#                     getattr(msg, "tool_call_id", None),
#                     getattr(msg, "function_call", None),
#                     getattr(msg, "reasoning", None),
#                     getattr(msg, "timestamp", None),
#                     getattr(msg, "metadata", None),
#                     getattr(msg, "usages", None),
#                     msg.raw if self.save_raw else None,
#                 )
#         return True

#     async def aget_message(self, config: dict[str, Any], message_id: str | int) -> Message:
#         """Retrieve a message by ID from the database."""
#         if not self._asyncpg_available or not self.pg_pool:
#             logger.error("asyncpg is not available or not initialized.")
#             raise IndexError("Message not found")
#         async with self.pg_pool.acquire() as conn:
#             row = await conn.fetchrow(
#                 "SELECT * FROM message_table WHERE message_id = $1", str(message_id)
#             )
#             if not row:
#                 raise IndexError(f"Message with ID {message_id} not found")
#             # Convert DB row to Message
#             return Message(
#                 message_id=row["message_id"],
#                 role=row["role"],
#                 content=row["content"],
#                 tools_calls=row["tools_calls"],
#                 tool_call_id=row["tool_call_id"],
#                 function_call=row["function_call"],
#                 reasoning=row["reasoning"],
#                 timestamp=row["timestamp"],
#                 metadata=row["metadata"] or {},
#                 usages=row["usages"],
#                 raw=row["raw"] if self.save_raw else None,
#             )

#     async def alist_messages(
#         self,
#         config: dict[str, Any],
#         search: str | None = None,
#         offset: int | None = None,
#         limit: int | None = None,
#     ) -> list[Message]:
#         """List messages for a thread, with optional search, offset, and limit."""
#         if not self._asyncpg_available or not self.pg_pool:
#             logger.error("asyncpg is not available or not initialized.")
#             return []
#         user_id = config.get("user")
#         thread_id = config.get("thread_id")
#         query = "SELECT * FROM message_table WHERE thread_id = $1"
#         params = [thread_id]
#         if user_id is not None:
#             query += " AND user_id = $2"
#             params.append(user_id)
#         if search:
#             query += " AND content ILIKE $%d" % (len(params) + 1)
#             params.append(f"%{search}%")
#         query += " ORDER BY timestamp ASC"
#         if limit:
#             query += f" LIMIT {limit}"
#         if offset:
#             query += f" OFFSET {offset}"
#         async with self.pg_pool.acquire() as conn:
#             rows = await conn.fetch(query, *params)
#             return [
#                 Message(
#                     message_id=row["message_id"],
#                     role=row["role"],
#                     content=row["content"],
#                     tools_calls=row["tools_calls"],
#                     tool_call_id=row["tool_call_id"],
#                     function_call=row["function_call"],
#                     reasoning=row["reasoning"],
#                     timestamp=row["timestamp"],
#                     metadata=row["metadata"] or {},
#                     usages=row["usages"],
#                     raw=row["raw"] if self.save_raw else None,
#                 )
#                 for row in rows
#             ]

#     async def adelete_message(self, config: dict[str, Any], message_id: str | int) -> Any:
#         """Delete a message by ID."""
#         if not self._asyncpg_available or not self.pg_pool:
#             logger.error("asyncpg is not available or not initialized.")
#             return False
#         async with self.pg_pool.acquire() as conn:
#             await conn.execute("DELETE FROM message_table WHERE message_id = $1", str(message_id))
#         return True

#     # -------------------------
#     # Message methods sync
#     # -------------------------
#     def put_messages(
#         self,
#         config: dict[str, Any],
#         messages: list[Message],
#         metadata: dict[str, Any] | None = None,
#     ) -> Any:
#         return asyncio.run(self.aput_messages(config, messages, metadata))

#     def get_message(self, config: dict[str, Any], message_id: str | int) -> Message:
#         return asyncio.run(self.aget_message(config, message_id))

#     def list_messages(
#         self,
#         config: dict[str, Any],
#         search: str | None = None,
#         offset: int | None = None,
#         limit: int | None = None,
#     ) -> list[Message]:
#         return asyncio.run(self.alist_messages(config, search, offset, limit))

#     def delete_message(self, config: dict[str, Any], message_id: str | int) -> Any:
#         return asyncio.run(self.adelete_message(config, message_id))

#     # -------------------------
#     # Thread methods async
#     # -------------------------
#     async def aput_thread(self, config: dict[str, Any], thread_info: dict[str, Any]) -> Any:
#         """Insert or update a thread in the database."""
#         if not self._asyncpg_available or not self.pg_pool:
#             logger.error("asyncpg is not available or not initialized.")
#             return False
#         user_id = config.get("user")
#         thread_id = thread_info.get("thread_id")
#         thread_name = thread_info.get("thread_name")
#         meta = thread_info.get("meta")
#         async with self.pg_pool.acquire() as conn:
#             await conn.execute(
#                 """
#                 INSERT INTO thread (thread_id, user_id, thread_name, meta, created_at, updated_at)
#                 VALUES ($1, $2, $3, $4, NOW(), NOW())
#                 ON CONFLICT (thread_id) DO UPDATE
#                 SET thread_name = $3, meta = $4, updated_at = NOW()
#                 """,
#                 thread_id,
#                 user_id,
#                 thread_name,
#                 meta,
#             )
#         return True

#     async def aget_thread(self, config: dict[str, Any]) -> Optional[dict[str, Any]]:
#         """Retrieve a thread by thread_id."""
#         if not self._asyncpg_available or not self.pg_pool:
#             logger.error("asyncpg is not available or not initialized.")
#             return None
#         thread_id = config.get("thread_id")
#         async with self.pg_pool.acquire() as conn:
#             row = await conn.fetchrow("SELECT * FROM thread WHERE thread_id = $1", thread_id)
#             if row:
#                 return dict(row)
#         return None

#     async def alist_threads(
#         self,
#         config: dict[str, Any],
#         search: str | None = None,
#         offset: int | None = None,
#         limit: int | None = None,
#     ) -> list[dict[str, Any]]:
#         """List threads, with optional search, offset, and limit."""
#         if not self._asyncpg_available or not self.pg_pool:
#             logger.error("asyncpg is not available or not initialized.")
#             return []
#         user_id = config.get("user")
#         query = "SELECT * FROM thread WHERE 1=1"
#         params = []
#         if user_id is not None:
#             query += " AND user_id = $1"
#             params.append(user_id)
#         if search:
#             query += f" AND thread_name ILIKE ${len(params) + 1}"
#             params.append(f"%{search}%")
#         query += " ORDER BY created_at ASC"
#         if limit:
#             query += f" LIMIT {limit}"
#         if offset:
#             query += f" OFFSET {offset}"
#         async with self.pg_pool.acquire() as conn:
#             rows = await conn.fetch(query, *params)
#             return [dict(row) for row in rows]

#     async def aclean_thread(self, config: dict[str, Any]) -> Any:
#         """Delete a thread by thread_id."""
#         if not self._asyncpg_available or not self.pg_pool:
#             logger.error("asyncpg is not available or not initialized.")
#             return False
#         thread_id = config.get("thread_id")
#         async with self.pg_pool.acquire() as conn:
#             await conn.execute("DELETE FROM thread WHERE thread_id = $1", thread_id)
#         return True

#     # -------------------------
#     # Thread methods sync
#     # -------------------------
#     def put_thread(self, config: dict[str, Any], thread_info: dict[str, Any]) -> Any:
#         return asyncio.run(self.aput_thread(config, thread_info))

#     def get_thread(self, config: dict[str, Any]) -> Optional[dict[str, Any]]:
#         return asyncio.run(self.aget_thread(config))

#     def list_threads(
#         self,
#         config: dict[str, Any],
#         search: str | None = None,
#         offset: int | None = None,
#         limit: int | None = None,
#     ) -> list[dict[str, Any]]:
#         return asyncio.run(self.alist_threads(config, search, offset, limit))

#     def clean_thread(self, config: dict[str, Any]) -> Any:
#         return asyncio.run(self.aclean_thread(config))

#     # -------------------------
#     # Clean Resources
#     # -------------------------
#     async def arelease(self) -> None:
#         """Close asyncpg and redis pools."""
#         if self.pg_pool:
#             await self.pg_pool.close()
#             self.pg_pool = None
#         if self.redis:
#             await self.redis.close()
#             self.redis = None
#         logger.info("PgRedisCheckpointer pools closed.")

#     def release(self) -> None:
#         """Sync resource cleanup."""
#         asyncio.run(self.arelease())
