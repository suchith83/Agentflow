# LLM Library Comparison: Finding the Best Alternative to LiteLLM

**Date**: January 8, 2026  
**Purpose**: Detailed technical comparison of LLM integration approaches

---

## Executive Summary

| Criteria | Winner | Rationale |
|----------|--------|-----------|
| **Reliability** | Official SDKs | First-party support, fastest bug fixes |
| **Control** | Official SDKs | No middleware, full access to features |
| **Privacy** | Official SDKs | Direct API calls, no third-party proxies |
| **Ease of Use** | OpenRouter | Single API, no multi-SDK management |
| **Cost** | Official SDKs | Free libraries, no additional fees |
| **Best for AgentFlow** | **Official SDKs** | Already have converter architecture |

---

## Option 1: Official Provider SDKs ⭐ RECOMMENDED

### Overview
Use official Python SDKs directly from each provider (OpenAI, Anthropic, Google, etc.) with AgentFlow's existing converter architecture.

### Architecture

```
User Code
    ↓
AgentFlow Agent Class (abstraction layer)
    ↓
ModelResponseConverter (routing layer)
    ↓
├─→ OpenAIConverter → openai SDK → OpenAI API
├─→ AnthropicConverter → anthropic SDK → Anthropic API
├─→ GoogleConverter → google-genai SDK → Google API
└─→ OtherConverters → other SDKs → Other APIs
```

### Supported Providers

| Provider | SDK | Models |
|----------|-----|--------|
| OpenAI | `openai>=1.50.0` | GPT-4o, GPT-4, o1, etc. |
| Anthropic | `anthropic>=0.40.0` | Claude 3.5 Sonnet, Opus, Haiku |
| Google | `google-genai>=1.56.0` | Gemini 2.0, Gemini Pro |
| Azure OpenAI | `openai>=1.50.0` | Same as OpenAI (via base_url) |
| Groq | `groq>=0.9.0` | Llama, Mixtral (OpenAI compatible) |
| Together AI | `together>=1.0.0` | Open source models |
| Cohere | `cohere>=5.0.0` | Command, Embed |
| Mistral | `mistralai>=1.0.0` | Mistral Large, Medium |

### Installation

```bash
# Install only what you need
pip install 10xscale-agentflow[openai]
pip install 10xscale-agentflow[anthropic]
pip install 10xscale-agentflow[google-genai]

# Or all at once
pip install 10xscale-agentflow[openai,anthropic,google-genai]
```

### Code Example

```python
from agentflow.graph import Agent

# Automatically detects provider from model name
openai_agent = Agent(model="gpt-4o", system_prompt="...")
anthropic_agent = Agent(model="claude-3-5-sonnet-20241022", system_prompt="...")
google_agent = Agent(model="gemini-2.0-flash-exp", system_prompt="...")

# All work the same way!
```

### Pros ✅

1. **Best Reliability**: First-party support, fastest bug fixes
2. **Latest Features**: Immediate access to new capabilities
3. **Best Documentation**: Official docs, examples, community
4. **Type Safety**: Full type hints, IDE autocomplete
5. **No Middleware Bugs**: Direct API calls, no translation layer issues
6. **Performance**: No additional latency from proxies
7. **Privacy**: Data goes directly to provider, no third parties
8. **Cost**: Free libraries, no subscription fees
9. **Flexibility**: Full control over all parameters
10. **Provider-Specific Features**: Access to unique capabilities
11. **Proven Architecture**: AgentFlow already has converters
12. **Selective Dependencies**: Users install only what they need

### Cons ⚠️

1. **Multiple Dependencies**: Need to manage several SDKs (but optional)
2. **Different APIs**: Each SDK has its own style (normalized by converters)
3. **More Converters**: Need to maintain converter for each provider (but isolated)
4. **Manual Provider Selection**: Need to specify which SDK to use (can auto-detect)

### When to Use

- ✅ You want maximum reliability and control
- ✅ You use a limited set of providers (1-3)
- ✅ You need provider-specific features
- ✅ You value privacy and data control
- ✅ You already have abstraction architecture (like AgentFlow!)

### Migration Complexity
**Low** - AgentFlow already has OpenAI and Google converters. Just add Anthropic.

---

## Option 2: OpenRouter

### Overview
API gateway service that provides access to 100+ LLMs through a single OpenAI-compatible endpoint.

### Architecture

```
User Code
    ↓
AgentFlow Agent Class
    ↓
OpenAI SDK (pointing to openrouter.ai)
    ↓
OpenRouter Gateway (EXTERNAL SERVICE)
    ↓
├─→ OpenAI API
├─→ Anthropic API
├─→ Google API
└─→ 100+ other providers
```

### Supported Providers
200+ models from OpenAI, Anthropic, Google, Meta, Mistral, and more.

### Installation

```bash
pip install 10xscale-agentflow[openai]
# No additional library needed - uses OpenAI SDK
```

### Code Example

```python
import openai

# Point OpenAI SDK to OpenRouter
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "sk-or-..."  # OpenRouter API key

from agentflow.graph import Agent

# Works with any model OpenRouter supports
agent = Agent(
    model="anthropic/claude-3-5-sonnet",  # OpenRouter format
    system_prompt="...",
)
```

### Pros ✅

1. **Single API**: One endpoint for all providers
2. **No SDK Management**: Just use OpenAI SDK
3. **Automatic Fallback**: If one provider is down, routes to another
4. **Cost Optimization**: Can choose cheapest provider automatically
5. **No Code Changes**: OpenAI-compatible API
6. **Wide Coverage**: 200+ models available
7. **Usage Analytics**: Built-in dashboards

### Cons ❌

1. **External Service**: Requires internet, can go down
2. **Latency**: Additional hop adds 50-200ms delay
3. **Privacy Concerns**: Your data passes through OpenRouter
4. **Vendor Lock-in**: Tied to OpenRouter's availability
5. **Limited Control**: Can't access all provider-specific features
6. **Cost**: Markup on API calls (though competitive)
7. **Rate Limits**: Subject to OpenRouter's limits
8. **Debugging**: Harder to troubleshoot (extra layer)

### When to Use

- ✅ You want maximum convenience
- ✅ You need to switch providers frequently
- ✅ You want automatic failover
- ✅ You don't mind third-party data handling
- ✅ Building SaaS where flexibility > control

### Migration Complexity
**Low** - Just change `api_base` in OpenAI SDK.

---

## Option 3: Portkey

### Overview
Enterprise-grade AI gateway with advanced observability, caching, and multi-provider routing.

### Architecture

```
User Code
    ↓
AgentFlow Agent Class
    ↓
Portkey SDK
    ↓
Portkey Gateway (can be self-hosted)
    ↓
├─→ OpenAI API
├─→ Anthropic API
├─→ Google API
└─→ 100+ other providers
```

### Supported Providers
1600+ LLMs via unified interface.

### Installation

```bash
pip install portkey-ai
pip install 10xscale-agentflow
```

### Code Example

```python
from portkey_ai import Portkey

client = Portkey(
    api_key="portkey_...",
    virtual_keys=["openai_...", "anthropic_..."]
)

# Use with AgentFlow (requires custom adapter)
# ... implementation details ...
```

### Pros ✅

1. **Enterprise Features**: Advanced observability, monitoring
2. **Automatic Retries**: Smart retry logic with exponential backoff
3. **Caching**: Reduce costs with semantic caching
4. **Load Balancing**: Distribute load across providers
5. **Self-Hostable**: Can run on your own infrastructure
6. **Cost Tracking**: Detailed analytics per request
7. **Fallback Routing**: Automatic failover to backup providers
8. **Security**: Enterprise-grade security features

### Cons ❌

1. **Complexity**: More setup and configuration
2. **Learning Curve**: Many features to understand
3. **Cost**: Paid plans for advanced features
4. **Overhead**: Additional infrastructure to maintain
5. **Overkill**: Too much for small projects
6. **Integration Work**: Need custom AgentFlow adapter
7. **External Dependency**: Another service to manage

### When to Use

- ✅ Enterprise deployment with thousands of users
- ✅ Need advanced observability and monitoring
- ✅ Want caching and cost optimization
- ✅ Require high availability with multiple providers
- ✅ Have dedicated DevOps resources

### Migration Complexity
**High** - Need to build custom Portkey adapter for AgentFlow.

---

## Option 4: any-llm

### Overview
Python library providing unified interface to multiple LLM providers, using official SDKs underneath.

### Architecture

```
User Code
    ↓
any-llm Library
    ↓
├─→ openai SDK → OpenAI API
├─→ anthropic SDK → Anthropic API
├─→ google-genai SDK → Google API
└─→ other SDKs → Other APIs
```

### Supported Providers
OpenAI, Anthropic, Google, Cohere, Groq, Together, Mistral.

### Installation

```bash
pip install any-llm
pip install 10xscale-agentflow
```

### Code Example

```python
from any_llm import Client

client = Client()

# Use same interface for all providers
response = await client.chat.completions.create(
    model="gpt-4o",  # or "claude-3-5-sonnet", "gemini-2.0-flash"
    messages=[...]
)
```

### Pros ✅

1. **Unified Interface**: Same API for all providers
2. **Uses Official SDKs**: Reliability of official libraries
3. **Type Hints**: Full IDE support
4. **Mozilla Backing**: Active maintenance and support
5. **Framework Agnostic**: Works with any Python framework
6. **No Proxy**: Direct API calls to providers

### Cons ❌

1. **Additional Abstraction**: Another layer to maintain
2. **Python 3.11+ Required**: Version constraint
3. **Dependency Bloat**: Installs multiple SDKs
4. **Less Mature**: Newer than official SDKs
5. **Maintenance Risk**: Depends on any-llm team
6. **Similar to LiteLLM**: Same category of problems
7. **Integration Work**: Need custom AgentFlow adapter
8. **Limited Documentation**: Smaller community

### When to Use

- ✅ You want unified interface but trust official SDKs
- ✅ You're okay with Python 3.11+ requirement
- ✅ You want to avoid writing converters yourself
- ❌ **NOT recommended for AgentFlow** - we already have converters!

### Migration Complexity
**Medium** - Need to build any-llm adapter for AgentFlow.

---

## Option 5: Continue with LiteLLM (Not Recommended)

### Current Issues

1. **Security**: CVE-2024-10188 (DoS vulnerability)
2. **Reliability**: Broken features, slow fixes
3. **Logging**: Excessive noise, hard to debug
4. **Maintenance**: Code quality concerns
5. **Lag**: Slower to support new features than official SDKs

### Why Not Recommended

LiteLLM's issues are structural, not fixable without major refactoring. Moving to official SDKs is more sustainable.

---

## Decision Matrix

### For AgentFlow Project Specifically

| Criteria | Weight | Official SDKs | OpenRouter | Portkey | any-llm |
|----------|--------|---------------|------------|---------|---------|
| Reliability | 5x | 5 = 25 | 4 = 20 | 4 = 20 | 3 = 15 |
| Control | 4x | 5 = 20 | 2 = 8 | 3 = 12 | 4 = 16 |
| Privacy | 4x | 5 = 20 | 2 = 8 | 4 = 16 | 5 = 20 |
| Ease of Use | 3x | 4 = 12 | 5 = 15 | 3 = 9 | 4 = 12 |
| Migration Cost | 3x | 5 = 15 | 4 = 12 | 2 = 6 | 3 = 9 |
| Maintenance | 4x | 5 = 20 | 5 = 20 | 3 = 12 | 3 = 12 |
| **Total** | | **112** | **83** | **75** | **84** |

**Winner: Official SDKs (112 points)**

---

## Recommendation for AgentFlow

### Primary Recommendation: Official Provider SDKs

**Rationale:**
1. AgentFlow **already has** converter architecture
2. Already have OpenAI and Google converters working
3. Just need to add Anthropic (and others as needed)
4. Best reliability and control
5. No new dependencies on external services
6. Privacy-friendly
7. Users install only what they need

### Implementation Path

1. **Phase 1** (Week 1): Add Anthropic converter
2. **Phase 2** (Week 2): Add deprecation warnings for LiteLLM
3. **Phase 3** (Month 2): Remove LiteLLM completely
4. **Phase 4** (Ongoing): Add more providers as needed

### Fallback Option: OpenRouter

**Use if:**
- User wants 100+ models without managing SDKs
- User prioritizes convenience over control
- User is building SaaS with frequent provider switching

**Implementation:**
- Already works with OpenAI converter (just change base URL)
- No code changes needed in AgentFlow

---

## Frequently Asked Questions

### Q: Why not just use OpenAI SDK for everything?

**A:** OpenAI SDK only works with OpenAI models. To support Anthropic Claude, Google Gemini, etc., you need their respective SDKs or a gateway service.

### Q: Why not use a unified library like LiteLLM or any-llm?

**A:** AgentFlow already has the abstraction layer (converters). Adding another unified library would be:
- Redundant (we already normalize responses)
- Less reliable (extra layer = extra bugs)
- Slower to get new features (depends on library updates)

### Q: What if I want to support 50+ models?

**A:** Then consider OpenRouter or Portkey. But most users only need 2-3 providers (OpenAI, Anthropic, Google), making official SDKs the better choice.

### Q: How hard is it to add a new provider?

**A:** With AgentFlow's architecture, adding a new provider is straightforward:
1. Create converter class (~300 lines)
2. Add to converter registry (~5 lines)
3. Add optional dependency (~1 line)
4. Write tests

Total time: ~1-2 days per provider.

### Q: What about cost?

**A:** Official SDKs are free. You only pay for API usage. OpenRouter and Portkey add markup or subscription fees.

### Q: Can I mix providers?

**A:** Yes! AgentFlow supports using different providers in the same graph:

```python
openai_agent = Agent(model="gpt-4o", ...)
anthropic_agent = Agent(model="claude-3-5-sonnet-20241022", ...)

graph.add_node("openai", openai_agent)
graph.add_node("anthropic", anthropic_agent)
```

---

## Conclusion

**For AgentFlow: Use Official Provider SDKs**

This approach:
- ✅ Builds on existing architecture
- ✅ Provides best reliability and control
- ✅ Respects user privacy
- ✅ Minimizes migration work
- ✅ Keeps dependencies optional
- ✅ Future-proof and maintainable

**Next Steps:**
1. Review this document with team
2. Implement Anthropic converter
3. Update documentation
4. Plan LiteLLM deprecation

---

**Questions? Contact AgentFlow Team**
