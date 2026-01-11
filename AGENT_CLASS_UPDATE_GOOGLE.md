# Agent Class Update: Fixed Google GenAI AsyncClient

**Date**: January 8, 2026  
**Fix**: Use Google GenAI's native `AsyncClient` instead of wrapping sync calls

---

## 🐛 Issue Found

Initial implementation incorrectly used synchronous `genai.Client()` and wrapped calls in `asyncio.to_thread()`.

## ✅ Fix Applied

Now correctly uses `genai.AsyncClient()` which provides native async methods:
- `await client.models.generate_content()` - Native async!
- `await client.models.generate_content_stream()` - Native async generator!

---

## 📝 Changes Made

### 1. Changed Client Creation

**Before** (Wrong):
```python
from google import genai
return genai.Client(api_key=api_key)  # Sync client
```

**After** (Correct):
```python
from google import genai
return genai.AsyncClient(api_key=api_key)  # Async client!
```

### 2. Removed asyncio.to_thread Wrapper

**Before** (Wrong):
```python
# Had to wrap sync calls
def _generate():
    return self.client.models.generate_content(...)

return await asyncio.to_thread(_generate)
```

**After** (Correct):
```python
# Native async, no wrapper needed!
return await self.client.models.generate_content(
    model=self.model,
    contents=google_contents,
    config=config,
)
```

### 3. Streaming Also Native Async

**Before** (Wrong):
```python
def _generate_stream():
    return self.client.models.generate_content_stream(...)

return await asyncio.to_thread(_generate_stream)  # Wrapped
```

**After** (Correct):
```python
# AsyncClient has async generator!
return await self.client.models.generate_content_stream(
    model=self.model,
    contents=google_contents,
    config=config,
)
```

---

## 🎯 Benefits

1. **True Async** - No blocking calls wrapped in threads
2. **Better Performance** - Native async is faster than thread pools
3. **Proper Async Generators** - Streaming works correctly
4. **Cleaner Code** - No wrapper functions needed

---

## 📚 Reference

- Google GenAI SDK Docs: [Synchronous and Asynchronous Operations](https://deepwiki.com/googleapis/python-genai/2.1.1-synchronous-and-asynchronous-operations)
- `genai.Client` - Synchronous client
- `genai.AsyncClient` - Asynchronous client (what we now use!)

---

## ✅ Status

**Fixed!** Google GenAI now uses native async throughout.
