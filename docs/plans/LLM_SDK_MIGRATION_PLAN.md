# LLM SDK Migration Plan: Replacing LiteLLM

**Date**: January 8, 2026  
**Status**: Research & Recommendation  
**Author**: AgentFlow Team

## Executive Summary

This document provides research findings and recommendations for replacing LiteLLM in the agentflow project due to identified issues with reliability, security, and maintenance.

**Recommendation**: **Use Official Provider SDKs with Native Converter Architecture**

---

## Table of Contents

1. [Background & Current Issues](#background--current-issues)
2. [Research Findings](#research-findings)
3. [Solution Comparison](#solution-comparison)
4. [Recommended Approach](#recommended-approach)
5. [Implementation Plan](#implementation-plan)
6. [Migration Strategy](#migration-strategy)
7. [Code Examples](#code-examples)

---

## Background & Current Issues

### Current Architecture

AgentFlow currently uses:
- **LiteLLM** for unified multi-provider LLM access
- **Custom converter architecture** in `agentflow/adapters/llm/`
- **Existing converters**: LiteLLMConverter, OpenAIConverter, GoogleGenAIConverter

### Problems with LiteLLM

1. **Security Vulnerabilities**
   - Critical CVE-2024-10188: DoS vulnerability via `ast.literal_eval`
   - Potential for resource exhaustion and service unavailability

2. **Code Quality & Maintenance**
   - Reports of broken features and delayed fixes
   - Internal architecture concerns
   - Excessive logging noise making debugging difficult

3. **Reliability Issues**
   - Inconsistent behavior across providers
   - Version conflicts and dependency bloat
   - Slower updates compared to official SDKs

---

## Research Findings

### Option 1: Official Provider SDKs ✅ **RECOMMENDED**

**Description**: Use native SDKs from each provider (OpenAI, Anthropic, Google, etc.) with agentflow's existing converter architecture.

**Pros:**
- ✅ Best-in-class support from each provider
- ✅ Fastest access to new features
- ✅ Most reliable and well-documented
- ✅ Better type safety and IDE support
- ✅ No middle-layer bugs or delays
- ✅ Leverages existing agentflow converter architecture
- ✅ Already have OpenAI and Google GenAI converters implemented

**Cons:**
- ⚠️ Need to maintain separate converters (already doing this!)
- ⚠️ Multiple dependencies (but optional via pyproject.toml)
- ⚠️ Different API styles (but normalized by converters)

**Supported Providers:**
- OpenAI (gpt-4, gpt-4o, o1, etc.)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, etc.)
- Google (Gemini 2.0, Gemini Pro, etc.)
- Cohere
- Mistral AI
- Together AI
- Groq (OpenAI-compatible API)
- Azure OpenAI (via OpenAI SDK)

### Option 2: OpenRouter

**Description**: Unified API gateway service providing access to 100+ LLMs through a single OpenAI-compatible endpoint.

**Pros:**
- ✅ Single API for multiple providers
- ✅ OpenAI SDK compatible
- ✅ Automatic failover and load balancing
- ✅ Cost optimization across providers
- ✅ No multi-SDK maintenance

**Cons:**
- ❌ Requires external service (not self-hosted)
- ❌ Additional latency from routing layer
- ❌ Potential privacy concerns (data passes through third party)
- ❌ Vendor lock-in to OpenRouter
- ❌ Limited control over provider-specific features

**Use Case**: Better for SaaS applications where convenience > control

### Option 3: Portkey

**Description**: Enterprise-grade AI gateway with observability and multi-provider support.

**Pros:**
- ✅ Advanced observability and monitoring
- ✅ Automatic retries and fallback routing
- ✅ Cost tracking per request
- ✅ Can be self-hosted

**Cons:**
- ❌ More complex setup
- ❌ Enterprise-focused (may be overkill)
- ❌ Additional infrastructure layer
- ❌ Cost for hosted version

**Use Case**: Better for large enterprise deployments

### Option 4: any-llm

**Description**: Python library providing unified interface using official provider SDKs.

**Pros:**
- ✅ Uses official SDKs underneath
- ✅ Full type hints
- ✅ Framework-agnostic
- ✅ Backed by Mozilla.ai

**Cons:**
- ❌ Requires Python 3.11+ (agentflow already uses 3.12+)
- ❌ Still an abstraction layer (similar to LiteLLM)
- ❌ Dependency bloat from multiple SDKs
- ❌ Less mature than official SDKs
- ❌ Additional maintenance risk

---

## Solution Comparison

| Feature | Official SDKs | OpenRouter | Portkey | any-llm | LiteLLM |
|---------|---------------|------------|---------|---------|---------|
| **Reliability** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Control** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Privacy** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Ease of Use** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Feature Access** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Maintenance** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Cost** | Free | Pay-per-use | Enterprise | Free | Free |
| **Self-hosted** | ✅ | ❌ | ✅ | ✅ | ✅ |

**Winner: Official SDKs** - Best balance of reliability, control, and feature access

---

## Recommended Approach

### Strategy: Multi-SDK Architecture with Native Converters

**Key Principles:**
1. Use official SDKs from each provider
2. Maintain converter architecture for normalization
3. Keep dependencies optional via extras
4. Provide clear migration path from LiteLLM

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 Agent Class (agent.py)              │
│           High-level abstraction for users          │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│          ModelResponseConverter                      │
│        (model_response_converter.py)                 │
│  Delegates to appropriate converter based on type   │
└─────────────┬───────────────────────────────────────┘
              │
              ├──────────┬──────────┬──────────┬───────┐
              ▼          ▼          ▼          ▼       ▼
     ┌─────────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐
     │   OpenAI    │ │ Anthropic│ │ Google │ │  Other  │
     │  Converter  │ │ Converter│ │Converter│ │Converter│
     └──────┬──────┘ └─────┬────┘ └────┬───┘ └────┬────┘
            │              │            │          │
            ▼              ▼            ▼          ▼
     ┌─────────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐
     │   openai    │ │anthropic │ │google  │ │  other  │
     │     SDK     │ │   SDK    │ │GenAI   │ │   SDKs  │
     └─────────────┘ └──────────┘ └────────┘ └─────────┘
```

### Benefits

1. **Reliability**: First-party support from each provider
2. **Flexibility**: Easy to add/remove providers
3. **Performance**: No unnecessary abstraction layers
4. **Maintainability**: Converters are isolated and testable
5. **User-friendly**: Users only install SDKs they need

---

## Implementation Plan

### Phase 1: Add Anthropic Support (Week 1)

**Priority: HIGH** - Anthropic's Claude is a major provider

#### Files to Create:
1. `agentflow/adapters/llm/anthropic_converter.py`
2. `agentflow/graph/anthropic_agent.py` (optional)

#### Updates Needed:
1. Update `agentflow/adapters/llm/__init__.py`
2. Update `agentflow/adapters/llm/model_response_converter.py`
3. Update `agentflow/adapters/llm/base_converter.py` (add ANTHROPIC enum)
4. Add to `pyproject.toml` optional dependencies

### Phase 2: Deprecate LiteLLM (Week 2)

1. Add deprecation warnings to LiteLLMConverter
2. Update documentation with migration guides
3. Update examples to use native SDKs
4. Create migration utilities

### Phase 3: Remove LiteLLM (Version 0.6.0)

1. Remove litellm from dependencies
2. Remove LiteLLMConverter
3. Remove litellm examples
4. Update changelog

### Phase 4: Enhance Native SDK Support (Ongoing)

1. Add converters for Cohere, Mistral, etc. (as needed)
2. Improve error handling
3. Add provider-specific features
4. Enhance documentation

---

## Migration Strategy

### For Existing Users

#### Before (LiteLLM):

```python
from agentflow.graph import Agent

agent = Agent(
    model="gpt-4",  # LiteLLM model string
    system_prompt="You are helpful",
    tools=[tool1, tool2],
)
```

#### After (Native OpenAI):

```python
from agentflow.graph import Agent

agent = Agent(
    model="gpt-4",  # Still works! Uses OpenAI SDK
    system_prompt="You are helpful",
    tools=[tool1, tool2],
    # Optional: specify converter explicitly
    converter="openai",  # Default for "gpt-*" models
)
```

### Migration Path

**Option A: Automatic (Recommended)**
- AgentFlow automatically detects model string and uses appropriate SDK
- `gpt-*` → OpenAI SDK
- `claude-*` → Anthropic SDK
- `gemini-*` → Google GenAI SDK

**Option B: Explicit**
- Users specify converter explicitly
- Provides more control
- Better for complex scenarios

### Breaking Changes

**None for most users!** 
- Model strings remain the same
- API remains backwards compatible
- Only breaking change: need to install provider SDKs

### Installation Changes

#### Before:
```bash
pip install 10xscale-agentflow[litellm]
```

#### After:
```bash
# Install only what you need
pip install 10xscale-agentflow[openai]
# or
pip install 10xscale-agentflow[anthropic]
# or
pip install 10xscale-agentflow[google-genai]
# or all
pip install 10xscale-agentflow[openai,anthropic,google-genai]
```

---

## Code Examples

### Example 1: OpenAI Agent

```python
from agentflow.graph import Agent, StateGraph
from agentflow.state import AgentState

# Create agent - automatically uses OpenAI SDK
agent = Agent(
    model="gpt-4o",
    system_prompt="You are a helpful assistant",
    tools=[calculator, weather],
)

# Use in graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.set_entry_point("agent")
```

### Example 2: Anthropic Agent

```python
from agentflow.graph import Agent

# Create agent - automatically uses Anthropic SDK
agent = Agent(
    model="claude-3-5-sonnet-20241022",
    system_prompt="You are a helpful assistant",
    tools=[calculator, weather],
)
```

### Example 3: Google Gemini Agent

```python
from agentflow.graph import Agent

# Create agent - automatically uses Google GenAI SDK
agent = Agent(
    model="gemini-2.0-flash-exp",
    system_prompt="You are a helpful assistant",
    tools=[calculator, weather],
)
```

### Example 4: Explicit Converter

```python
from agentflow.graph import Agent
from openai import AsyncOpenAI

# For advanced use cases, use SDK directly
client = AsyncOpenAI(api_key="...")

async def custom_agent(state, config):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=state.context,
    )
    
    # Convert using agentflow converter
    from agentflow.adapters.llm import OpenAIConverter
    converter = OpenAIConverter()
    message = await converter.convert_response(response)
    
    return {"context": [message]}
```

### Example 5: Multi-Provider Graph

```python
from agentflow.graph import Agent, StateGraph
from agentflow.state import AgentState

# Different providers in same graph!
openai_agent = Agent(model="gpt-4o", system_prompt="...")
anthropic_agent = Agent(model="claude-3-5-sonnet-20241022", system_prompt="...")
google_agent = Agent(model="gemini-2.0-flash-exp", system_prompt="...")

graph = StateGraph(AgentState)
graph.add_node("openai", openai_agent)
graph.add_node("anthropic", anthropic_agent)
graph.add_node("google", google_agent)

# Route based on task complexity
def route(state):
    if state.task_complexity == "simple":
        return "google"  # Fast and cheap
    elif state.task_complexity == "medium":
        return "openai"
    else:
        return "anthropic"  # Best reasoning

graph.add_conditional_edges("START", route)
```

---

## Next Steps

### Immediate Actions (This Week)

1. ✅ **Review this document** - Get team alignment
2. ⬜ **Create Anthropic converter** - Fill the major gap
3. ⬜ **Update documentation** - Migration guides
4. ⬜ **Add deprecation warnings** - Warn LiteLLM users

### Short Term (This Month)

1. ⬜ **Update all examples** - Use native SDKs
2. ⬜ **Add integration tests** - For each converter
3. ⬜ **Create migration utilities** - Help users transition
4. ⬜ **Update tutorials** - Reflect new approach

### Long Term (Next Quarter)

1. ⬜ **Remove LiteLLM** - Version 0.6.0 release
2. ⬜ **Add more converters** - Cohere, Mistral, etc.
3. ⬜ **Enhanced features** - Provider-specific capabilities
4. ⬜ **Performance optimization** - Benchmarking and tuning

---

## Appendix: Technical Details

### Provider SDK Versions

```toml
[project.optional-dependencies]
openai = ["openai>=1.50.0"]
anthropic = ["anthropic>=0.40.0"]
google-genai = ["google-genai>=1.56.0"]
cohere = ["cohere>=5.0.0"]
mistral = ["mistralai>=1.0.0"]

# Legacy (deprecated)
litellm = ["litellm>=1.77.0"]
```

### Model String Detection

```python
MODEL_PREFIXES = {
    "gpt-": "openai",
    "o1-": "openai",
    "claude-": "anthropic",
    "gemini-": "google",
    "command-": "cohere",
    "mistral-": "mistral",
}

def detect_provider(model: str) -> str:
    for prefix, provider in MODEL_PREFIXES.items():
        if model.startswith(prefix):
            return provider
    return "openai"  # Default fallback
```

### Converter Registry

```python
CONVERTER_REGISTRY = {
    "openai": OpenAIConverter,
    "anthropic": AnthropicConverter,
    "google": GoogleGenAIConverter,
    "litellm": LiteLLMConverter,  # Deprecated
}
```

---

## Conclusion

**Recommendation**: **Adopt Official Provider SDKs with Native Converters**

This approach provides:
- ✅ **Best reliability** - First-party support
- ✅ **Maximum control** - No middleware dependencies
- ✅ **Future-proof** - Easy to add new providers
- ✅ **User-friendly** - Minimal breaking changes
- ✅ **Maintainable** - Clean architecture

The existing converter architecture in agentflow already provides the abstraction needed. We just need to complete the set of converters and deprecate LiteLLM.

---

**Questions? Contact the AgentFlow team.**
