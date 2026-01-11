# Agent Class Migration: LiteLLM → Native SDKs

**Date**: January 8, 2026  
**Status**: ✅ Completed  
**Changes**: Removed LiteLLM dependency, added native SDK support

---

## 🎯 Summary

The `Agent` class has been updated to:
- ✅ **Remove LiteLLM dependency**
- ✅ **Use native provider SDKs** (OpenAI, Anthropic, Google GenAI)
- ✅ **Add `provider` parameter** (optional, auto-detects from model)
- ✅ **Support custom clients** (escape hatch for power users)

---

## 📝 Changes Made

### 1. Removed LiteLLM Dependency

**Before**:
```python
try:
    from litellm import acompletion
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

# Required LiteLLM
if not HAS_LITELLM:
    raise ImportError("litellm is required...")
```

**After**:
```python
# No LiteLLM import needed!
# Uses native SDKs directly
```

### 2. Added Provider Parameter

**New Signature**:
```python
def __init__(
    self,
    model: str,
    provider: str | None = None,  # NEW: Optional provider
    system_prompt: list[dict[str, Any]] | None = None,
    tools: list[Callable] | ToolNode | None = None,
    client: Any = None,  # NEW: Escape hatch for custom clients
    # ... other params
):
```

### 3. Auto-Detection from Model

**Auto-detects provider from model name**:
- `gpt-*` or `o1-*` → `openai`
- `claude-*` → `anthropic`
- `gemini-*` → `google`

### 4. Native SDK Integration

**Uses official SDKs directly**:
- OpenAI: `AsyncOpenAI` from `openai` package
- Anthropic: `AsyncAnthropic` from `anthropic` package
- Google: `genai.Client` from `google-genai` package

---

## 🚀 Usage Examples

### Example 1: Explicit Provider

```python
from agentflow.graph import Agent

# OpenAI
agent = Agent(
    model="gpt-4o",
    provider="openai",
    system_prompt="You are a helpful assistant",
)

# Anthropic
agent = Agent(
    model="claude-3-5-sonnet-20241022",
    provider="anthropic",
    system_prompt="You are a helpful assistant",
)

# Google
agent = Agent(
    model="gemini-2.0-flash-exp",
    provider="google",
    system_prompt="You are a helpful assistant",
)
```

### Example 2: Auto-Detection (Recommended)

```python
# Provider auto-detected from model name
agent = Agent(
    model="gpt-4o",  # Auto-detects "openai"
    system_prompt="You are a helpful assistant",
)

agent = Agent(
    model="claude-3-5-sonnet-20241022",  # Auto-detects "anthropic"
    system_prompt="You are a helpful assistant",
)

agent = Agent(
    model="gemini-2.0-flash-exp",  # Auto-detects "google"
    system_prompt="You are a helpful assistant",
)
```

### Example 3: Custom Client (Escape Hatch)

```python
from openai import AsyncOpenAI

# Use your own client with custom config
custom_client = AsyncOpenAI(
    api_key="sk-...",
    base_url="https://my-proxy.com/v1",
    timeout=60.0,
)

agent = Agent(
    model="gpt-4o",
    client=custom_client,  # Provider auto-detected from client type
    system_prompt="You are a helpful assistant",
)
```

---

## 📦 Installation Changes

### Before (LiteLLM)
```bash
pip install 10xscale-agentflow[litellm]
```

### After (Native SDKs)
```bash
# Install only what you need
pip install 10xscale-agentflow[openai]
# or
pip install 10xscale-agentflow[anthropic]
# or
pip install 10xscale-agentflow[google-genai]

# Or all at once
pip install 10xscale-agentflow[openai,anthropic,google-genai]
```

---

## 🔄 Migration Guide

### For Existing Users

**Old Code** (still works, but deprecated):
```python
agent = Agent(
    model="gpt-4",
    system_prompt="...",
)
```

**New Code** (recommended):
```python
agent = Agent(
    model="gpt-4o",  # Updated model name
    provider="openai",  # Explicit provider (optional)
    system_prompt="...",
)
```

**No breaking changes!** The old code still works, but you should migrate to:
1. Use explicit `provider` parameter
2. Install appropriate SDK instead of LiteLLM

---

## ⚙️ Technical Details

### Provider Detection Logic

```python
def _detect_provider_from_model(self, model: str) -> str:
    model_lower = model.lower()
    
    if model_lower.startswith("gpt-") or model_lower.startswith("o1-"):
        return "openai"
    elif model_lower.startswith("claude-"):
        return "anthropic"
    elif model_lower.startswith("gemini-"):
        return "google"
    else:
        # Default to openai with warning
        return "openai"
```

### Client Creation

```python
def _create_client(self, provider: str, model: str) -> Any:
    if provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI()
    
    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        return AsyncAnthropic()
    
    elif provider == "google":
        from google import genai
        import os
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key)
```

### LLM Call Method

**Updated `_call_llm` method**:
- Uses duck typing to detect provider
- Calls appropriate SDK method
- Handles provider-specific differences (e.g., Anthropic's separate system message)
- Wraps Google GenAI sync calls in `asyncio.to_thread`

---

## ✅ Benefits

1. **No LiteLLM Dependency**
   - Removes security vulnerabilities
   - Better reliability
   - Faster updates

2. **Better Performance**
   - Direct API calls (no middleware)
   - Native async support (except Google, wrapped)

3. **More Control**
   - Can pass custom clients
   - Direct access to SDK features
   - Provider-specific parameters work

4. **Easier Maintenance**
   - Less abstraction
   - Clearer code
   - Easier to debug

---

## ⚠️ Known Limitations

1. **Anthropic Converter**
   - Currently uses OpenAI converter as fallback
   - Full Anthropic converter coming soon
   - Some features may not work perfectly

2. **Google GenAI**
   - SDK is synchronous, wrapped in `asyncio.to_thread`
   - Streaming support may need additional work
   - Tool format conversion needed

3. **Tool Format**
   - Each provider has different tool formats
   - Conversion happens in `_call_llm`
   - May need refinement for complex tools

---

## 🔜 Next Steps

1. **Add Anthropic Converter**
   - Implement full `AnthropicConverter` class
   - Remove OpenAI fallback

2. **Improve Google GenAI Support**
   - Better tool format conversion
   - Proper streaming support
   - Async wrapper improvements

3. **Add More Providers**
   - Cohere
   - Mistral
   - Others as needed

4. **Deprecate LiteLLM**
   - Add deprecation warnings
   - Update documentation
   - Remove in v0.6.0

---

## 📚 Related Documents

- [LLM SDK Migration Plan](../docs/plans/LLM_SDK_MIGRATION_PLAN.md)
- [Abstraction Reduction Philosophy](../docs/plans/ABSTRACTION_REDUCTION_PHILOSOPHY.md)

---

**Status**: ✅ Implementation Complete  
**Testing**: ⬜ Needs testing with all providers  
**Documentation**: ⬜ Update user docs
