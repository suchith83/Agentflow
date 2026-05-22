import base64
from unittest.mock import AsyncMock, patch

import pytest

from agentflow.core.exceptions.media_exceptions import UnsupportedMediaInputError
from agentflow.core.state.message_block import MediaRef
from agentflow.storage.media.capabilities import MediaTransportMode
from agentflow.storage.media.media_resolver import MediaResolver, _openai_image_url


class _Store:
    def __init__(self):
        self.retrieve_map = {"k": (b"abc", "image/png")}
        self.url_map = {"k": "https://signed.example/k.png"}

    async def retrieve(self, key):
        return self.retrieve_map[key]

    async def get_direct_url(self, key, mime_type=None):
        return self.url_map.get(key)


class _Cache:
    def __init__(self):
        self.values = {}
        self.put_calls = []

    async def aget_cache_value(self, namespace, key):
        return self.values.get((namespace, key))

    async def aput_cache_value(self, namespace, key, value, ttl_seconds=0):
        self.values[(namespace, key)] = value
        self.put_calls.append((namespace, key, value, ttl_seconds))


@pytest.mark.asyncio
async def test_try_transport_returns_none_for_unsupported_mode():
    resolver = MediaResolver()
    ref = MediaRef(kind="url", url="https://example.com/x.png")
    result = await resolver._try_transport(ref, MediaTransportMode.unsupported, "openai", "gpt-4o", object())
    assert result is None


@pytest.mark.asyncio
async def test_transport_remote_url_returns_none_for_internal_ref_without_store():
    resolver = MediaResolver(media_store=None)
    ref = MediaRef(kind="url", url="agentflow://media/k")
    result = await resolver._transport_remote_url(ref, type("Caps", (), {"accepts_external_urls": True})())
    assert result is None


@pytest.mark.asyncio
async def test_transport_remote_url_respects_external_url_capability_flag():
    resolver = MediaResolver()
    ref = MediaRef(kind="url", url="https://example.com/x.png")
    result = await resolver._transport_remote_url(
        ref,
        type("Caps", (), {"accepts_external_urls": False})(),
    )
    assert result is None


@pytest.mark.asyncio
async def test_retrieve_bytes_from_data_ref():
    payload = base64.b64encode(b"hello").decode()
    resolver = MediaResolver()
    data, mime = await resolver._retrieve_bytes(MediaRef(kind="data", data_base64=payload, mime_type="text/plain"))
    assert data == b"hello"
    assert mime == "text/plain"


@pytest.mark.asyncio
async def test_fetch_external_url_reads_response_bytes_and_mime():
    resolver = MediaResolver()

    class _Resp:
        headers = {"Content-Type": "image/jpeg"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        async def read(self):
            return b"jpg"

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return _Resp()

    with patch("aiohttp.ClientSession", return_value=_Session()):
        data, mime = await resolver._fetch_external_url("https://x")

    assert data == b"jpg"
    assert mime == "image/jpeg"


@pytest.mark.asyncio
async def test_get_direct_url_uses_cache_hit_when_not_expiring(monkeypatch):
    store = _Store()
    cache = _Cache()
    resolver = MediaResolver(media_store=store, cache_backend=cache)

    cache.values[("media:signed-url", "k:application/octet-stream:3600")] = {
        "url": "https://cached.example/k.png",
        "expires_at": 9999999999,
    }

    result = await resolver._get_direct_url(MediaRef(kind="url", url="agentflow://media/k"))
    assert result == "https://cached.example/k.png"


@pytest.mark.asyncio
async def test_get_direct_url_puts_cache_on_miss():
    store = _Store()
    cache = _Cache()
    resolver = MediaResolver(media_store=store, cache_backend=cache)

    result = await resolver._get_direct_url(MediaRef(kind="url", url="agentflow://media/k", mime_type="image/png"))
    assert result == "https://signed.example/k.png"
    assert len(cache.put_calls) == 1


@pytest.mark.asyncio
async def test_resolve_raises_with_attempted_transports_when_all_fail(monkeypatch):
    resolver = MediaResolver()
    ref = MediaRef(kind="url", url="https://bad.example/fail.png")

    async def _always_fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(resolver, "_try_transport", _always_fail)

    with pytest.raises(UnsupportedMediaInputError) as exc:
        await resolver.resolve(ref, provider="openai", model="gpt-4o", media_type="image")

    assert "All transports failed" in str(exc.value)


@pytest.mark.asyncio
async def test_transport_provider_file_handles_google_uri_and_data(monkeypatch):
    resolver = MediaResolver(media_store=_Store())

    class _Part:
        @staticmethod
        def from_uri(file_uri, mime_type):
            return {"uri": file_uri, "mime": mime_type}

    async def _upload(data, mime):
        return {"uploaded": True, "mime": mime, "size": len(data)}

    with patch("google.genai.types.Part", _Part), patch(
        "agentflow.storage.media.provider_media.upload_to_google_file_api",
        new=_upload,
    ):
        gs = await resolver._transport_provider_file(
            MediaRef(kind="url", url="gs://bucket/x.jpg", mime_type="image/jpeg"),
            provider="google",
            model="gemini-1.5-pro",
        )
        assert gs["uri"].startswith("gs://")

        data_ref = await resolver._transport_provider_file(
            MediaRef(kind="data", data_base64=base64.b64encode(b"abc").decode(), mime_type="image/png"),
            provider="google",
            model="gemini-1.5-pro",
        )
        assert data_ref["uploaded"] is True


@pytest.mark.asyncio
async def test_transport_provider_file_handles_internal_and_external_url_uploads():
    resolver = MediaResolver(media_store=_Store())

    async def _upload(data, mime):
        return {"uploaded": True, "mime": mime, "size": len(data)}

    with patch("agentflow.storage.media.provider_media.upload_to_google_file_api", new=_upload), patch.object(
        resolver,
        "_retrieve_bytes",
        new=AsyncMock(side_effect=[(b"in", "image/png"), (b"out", "image/jpeg")]),
    ):
        a = await resolver._transport_provider_file(
            MediaRef(kind="url", url="agentflow://media/k"),
            provider="google",
            model="gemini-1.5-pro",
        )
        b = await resolver._transport_provider_file(
            MediaRef(kind="url", url="https://example.com/x.jpg"),
            provider="google",
            model="gemini-1.5-pro",
        )

    assert a["uploaded"] is True
    assert b["uploaded"] is True


@pytest.mark.asyncio
async def test_transport_provider_file_returns_none_on_non_google_and_errors(monkeypatch):
    resolver = MediaResolver(media_store=_Store())
    assert (
        await resolver._transport_provider_file(
            MediaRef(kind="url", url="https://x"),
            provider="openai",
            model="gpt-4o",
        )
        is None
    )

    async def _broken(*args, **kwargs):
        raise RuntimeError("upload failed")

    with patch("agentflow.storage.media.provider_media.upload_to_google_file_api", new=_broken):
        out = await resolver._transport_provider_file(
            MediaRef(kind="data", data_base64=base64.b64encode(b"abc").decode(), mime_type="image/png"),
            provider="google",
            model="gemini-1.5-pro",
        )
    assert out is None


@pytest.mark.asyncio
async def test_transport_provider_file_returns_none_for_unhandled_google_ref_kind():
    resolver = MediaResolver(media_store=_Store())
    out = await resolver._transport_provider_file(
        MediaRef(kind="file_id", file_id="f-1"),
        provider="google",
        model="gemini-1.5-pro",
    )
    assert out is None


@pytest.mark.asyncio
async def test_retrieve_and_get_direct_url_none_paths():
    resolver = MediaResolver(media_store=None, cache_backend=None)

    with pytest.raises(RuntimeError):
        await resolver._retrieve("agentflow://media/missing")

    assert await resolver._get_direct_url(MediaRef(kind="url", url=None)) is None


@pytest.mark.asyncio
async def test_get_direct_url_ignores_invalid_cache_payload_and_falls_back_to_store():
    store = _Store()
    cache = _Cache()
    resolver = MediaResolver(media_store=store, cache_backend=cache)
    cache.values[("media:signed-url", "k:image/png:3600")] = {
        "url": 123,
        "expires_at": "not-a-number",
    }

    out = await resolver._get_direct_url(MediaRef(kind="url", url="agentflow://media/k", mime_type="image/png"))
    assert out == "https://signed.example/k.png"


@pytest.mark.asyncio
async def test_get_direct_url_returns_none_when_store_has_no_url():
    store = _Store()
    store.url_map["k"] = None
    resolver = MediaResolver(media_store=store, cache_backend=None)
    out = await resolver._get_direct_url(MediaRef(kind="url", url="agentflow://media/k", mime_type="image/png"))
    assert out is None


@pytest.mark.asyncio
async def test_retrieve_bytes_invalid_kind_raises():
    resolver = MediaResolver(media_store=_Store())
    with pytest.raises(ValueError):
        await resolver._retrieve_bytes(MediaRef(kind="file_id", file_id="x"))


def test_openai_image_url_helper_shape():
    part = _openai_image_url("https://example.com/x.png")
    assert part == {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
