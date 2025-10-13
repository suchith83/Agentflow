"""
Integration adapters for optional third-party SDKs.

This module exposes unified wrappers for integrating external tool registries
and SDKs with agentflow agent graphs. The adapters provide registry-based
discovery, function-calling schemas, and normalized execution for supported
tool providers.

Exports:
        ComposioAdapter: Adapter for the Composio Python SDK.
        LangChainAdapter: Adapter for LangChain tool registry and execution.
"""

from .composio_adapter import ComposioAdapter
from .langchain_adapter import LangChainAdapter


__all__ = ["ComposioAdapter", "LangChainAdapter"]
