import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentflow.storage.checkpointer.pg_checkpointer import PgCheckpointer
from agentflow.core.state import AgentState


@pytest.fixture
def cp(monkeypatch):
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_ASYNCPG", True)
    monkeypatch.setattr("agentflow.storage.checkpointer.pg_checkpointer.HAS_REDIS", True)

    class _Redis:
        def __init__(self):
            self.setex = AsyncMock()
            self.get = AsyncMock(return_value=None)
            self.delete = AsyncMock(return_value=1)
            self.scan = AsyncMock(side_effect=[(1, [b"generic_cache:ns:a"]), (0, [b"generic_cache:ns:b"])])

    redis = _Redis()
    c = PgCheckpointer(postgres_dsn="postgres://x", redis=redis)
    c._pg_pool = MagicMock()
    return c


def test_table_and_schema_name_validation(cp):
    assert cp._get_table_name("threads") == '"public"."threads"'
    with pytest.raises(ValueError):
        cp._get_table_name("bad-name")


@pytest.mark.asyncio
async def test_generic_cache_methods(cp):
    assert await cp.aput_cache_value("ns", "k", {"x": 1}, ttl_seconds=12) is True
    cp.redis.get.return_value = json.dumps({"x": 1})
    assert await cp.aget_cache_value("ns", "k") == {"x": 1}
    assert await cp.aclear_cache_value("ns", "k") == 1
    keys = await cp.alist_cache_keys("ns")
    assert keys == ["a", "b"]


@pytest.mark.asyncio
async def test_generic_cache_failures_return_safe_values(cp):
    cp.redis.setex.side_effect = RuntimeError("x")
    cp.redis.get.side_effect = RuntimeError("x")
    cp.redis.delete.side_effect = RuntimeError("x")
    cp.redis.scan.side_effect = RuntimeError("x")

    assert await cp.aput_cache_value("ns", "k", {"x": 1}) is None
    assert await cp.aget_cache_value("ns", "k") is None
    assert await cp.aclear_cache_value("ns", "k") is None
    assert await cp.alist_cache_keys("ns") == []


def test_validate_config_errors(cp):
    with pytest.raises(ValueError):
        cp._validate_config({"thread_id": "t"})
    with pytest.raises(ValueError):
        cp._validate_config({"user_id": "u"})


def test_json_serializer_fallback(monkeypatch, cp):
    monkeypatch.setenv("FAST_JSON", "1")
    assert callable(cp._get_json_serializer())


def test_deserialize_state_from_bytes_and_dict(cp):
    state = AgentState()
    payload = cp._serialize_state(state).encode()
    out1 = cp._deserialize_state(payload, AgentState)
    assert isinstance(out1, AgentState)

    out2 = cp._deserialize_state(state.model_dump(), AgentState)
    assert isinstance(out2, AgentState)


def test_row_to_message_handles_string_and_bytes_json(cp):
    row = {
        "message_id": "m1",
        "role": "assistant",
        "content": "plain text",
        "tool_calls": json.dumps([{"name": "x"}]),
        "tool_call_id": None,
        "reasoning": "r",
        "created_at": None,
        "total_tokens": 0,
        "usages": json.dumps({"completion_tokens": 1, "prompt_tokens": 2, "total_tokens": 3}),
        "meta": json.dumps({"k": "v"}),
    }
    msg = cp._row_to_message(row)
    assert msg.role == "assistant"
    assert msg.metadata["k"] == "v"

    row2 = dict(row)
    row2["content"] = b"{\"type\": \"text\", \"text\": \"hi\"}"
    row2["tool_calls"] = b"invalid-json"
    row2["usages"] = b"invalid-json"
    row2["meta"] = b"invalid-json"
    msg2 = cp._row_to_message(row2)
    assert msg2.role == "assistant"


class _AcquireCtx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_message_methods_cover_query_paths(cp):
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(
        return_value={
            "message_id": "m1",
            "role": "assistant",
            "content": "hello",
            "tool_calls": None,
            "tool_call_id": None,
            "reasoning": None,
            "created_at": None,
            "total_tokens": 0,
            "usages": None,
            "meta": None,
        }
    )
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock()

    cp._pg_pool = MagicMock()
    cp._pg_pool.acquire.return_value = _AcquireCtx(conn)

    out = await cp.aget_message({"thread_id": "t1"}, "m1")
    assert out.message_id == "m1"

    rows = await cp.alist_messages({"thread_id": "t1"}, search="hello", offset=1, limit=2)
    assert rows == []

    await cp.adelete_message({"thread_id": "t1"}, "m1")
    await cp.adelete_message({}, "m1")
    assert conn.execute.await_count >= 2


@pytest.mark.asyncio
async def test_thread_methods_cover_insert_update_and_list(cp):
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(side_effect=[{"thread_id": "t1"}, None, None, {"meta": json.dumps({"run_id": "r1"}), "thread_name": "T", "updated_at": None}])
    conn.fetch = AsyncMock(
        return_value=[
            {
                "thread_id": "t1",
                "thread_name": "T",
                "user_id": "u1",
                "created_at": None,
                "updated_at": None,
                "meta": json.dumps({"run_id": "r1"}),
            }
        ]
    )
    conn.execute = AsyncMock()

    cp._pg_pool = MagicMock()
    cp._pg_pool.acquire.return_value = _AcquireCtx(conn)

    from agentflow.utils.thread_info import ThreadInfo

    created = await cp.aput_thread({"thread_id": "t1", "user_id": "u1"}, ThreadInfo(thread_id="t1", thread_name="T"))
    assert created is True

    created2 = await cp.aput_thread(
        {"thread_id": "t1", "user_id": "u1"},
        ThreadInfo(thread_id="t1", thread_name=None, metadata={"x": 1}),
    )
    assert created2 is False

    thr = await cp.aget_thread({"thread_id": "t1", "user_id": "u1"})
    assert thr is None or thr.thread_id == "t1"

    listed = await cp.alist_threads({"user_id": "u1"}, search="T", offset=1, limit=1)
    assert len(listed) == 1


@pytest.mark.asyncio
async def test_aget_message_not_found_raises(cp):
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    cp._pg_pool = MagicMock()
    cp._pg_pool.acquire.return_value = _AcquireCtx(conn)

    with pytest.raises(ValueError):
        await cp.aget_message({"thread_id": "t1"}, "missing")
