"""Composio adapter for agentflow.

This module provides a thin wrapper around the Composio Python SDK to:
- Fetch tools formatted for LLM function calling (matching ToolNode format)
- Execute Composio tools directly

The dependency is optional. Install with:
    pip install 10xscale-agentflow[composio]

Usage outline:
    adapter = ComposioAdapter(api_key=os.environ["COMPOSIO_API_KEY"])  # optional key
    result = adapter.execute(
        slug="GITHUB_LIST_STARGAZERS",
        arguments={"owner": "ComposioHQ", "repo": "composio"},
        user_id="user-123",
    )
"""

from __future__ import annotations

import logging
import typing as t


logger = logging.getLogger(__name__)


try:
    # Lazy import composio SDK
    from composio import Composio  # type: ignore

    HAS_COMPOSIO = True
except Exception:  # ImportError or other
    Composio = None  # type: ignore
    HAS_COMPOSIO = False


class ComposioAdapter:
    """Adapter around Composio Python SDK.

    Notes on SDK methods used (from docs):
    - composio.tools.get(user_id=..., tools=[...]/toolkits=[...]/search=..., scopes=..., limit=...)
        Returns tools formatted for providers or agent frameworks; includes schema.
    - composio.tools.get_raw_composio_tools(...)
        Returns raw tool schemas including input_parameters.
    - composio.tools.execute(slug, arguments, user_id=..., connected_account_id=..., ...)
        Executes a tool and returns a dict like {data, successful, error}.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        provider: t.Any | None = None,
        file_download_dir: str | None = None,
        toolkit_versions: t.Any | None = None,
    ) -> None:
        """
        Initialize the ComposioAdapter.

        Args:
            api_key (str | None): Optional API key for Composio.
            provider (Any | None): Optional provider integration.
            file_download_dir (str | None): Directory for auto file handling.
            toolkit_versions (Any | None): Toolkit version overrides.

        Raises:
            ImportError: If composio SDK is not installed.
        """
        if not HAS_COMPOSIO:
            raise ImportError(
                "ComposioAdapter requires 'composio' package. Install with: "
                "pip install 10xscale-agentflow[composio]"
            )

        self._composio = Composio(  # type: ignore[call-arg]
            api_key=api_key,
            provider=provider,
            file_download_dir=file_download_dir,
            toolkit_versions=toolkit_versions,
        )

    @staticmethod
    def is_available() -> bool:
        """
        Return True if composio SDK is importable.

        Returns:
            bool: True if composio SDK is available, False otherwise.
        """
        return HAS_COMPOSIO

    def list_tools_for_llm(
        self,
        *,
        user_id: str,
        tool_slugs: list[str] | None = None,
        toolkits: list[str] | None = None,
        search: str | None = None,
        scopes: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, t.Any]]:
        """
        Return tools formatted for LLM function-calling.

        Args:
            user_id (str): User ID for tool discovery.
            tool_slugs (list[str] | None): Optional list of tool slugs.
            toolkits (list[str] | None): Optional list of toolkits.
            search (str | None): Optional search string.
            scopes (list[str] | None): Optional scopes.
            limit (int | None): Optional limit on number of tools.

        Returns:
            list[dict[str, Any]]: List of tools in function-calling format.
        """
        # Prefer the provider-wrapped format when available
        tools = self._composio.tools.get(
            user_id=user_id,
            tools=tool_slugs,  # type: ignore[arg-type]
            toolkits=toolkits,  # type: ignore[arg-type]
            search=search,
            scopes=scopes,
            limit=limit,
        )

        # The provider-wrapped output may already be in the desired structure.
        # We'll detect and pass-through; otherwise convert using raw schemas.
        formatted: list[dict[str, t.Any]] = []
        for t_obj in tools if isinstance(tools, list) else []:
            try:
                if (
                    isinstance(t_obj, dict)
                    and t_obj.get("type") == "function"
                    and "function" in t_obj
                ):
                    formatted.append(t_obj)
                else:
                    # Fallback: try to pull minimal fields
                    fn = t_obj.get("function", {}) if isinstance(t_obj, dict) else {}
                    if fn.get("name") and fn.get("parameters"):
                        formatted.append({"type": "function", "function": fn})
            except Exception as exc:
                logger.debug("Skipping non-conforming Composio tool wrapper: %s", exc)
                continue

        if formatted:
            return formatted

        # Fallback to raw schemas and convert manually
        formatted.extend(
            self.list_raw_tools_for_llm(
                tool_slugs=tool_slugs, toolkits=toolkits, search=search, scopes=scopes, limit=limit
            )
        )

        return formatted

    def list_raw_tools_for_llm(
        self,
        *,
        tool_slugs: list[str] | None = None,
        toolkits: list[str] | None = None,
        search: str | None = None,
        scopes: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, t.Any]]:
        """
        Return raw Composio tool schemas mapped to function-calling format.

        Args:
            tool_slugs (list[str] | None): Optional list of tool slugs.
            toolkits (list[str] | None): Optional list of toolkits.
            search (str | None): Optional search string.
            scopes (list[str] | None): Optional scopes.
            limit (int | None): Optional limit on number of tools.

        Returns:
            list[dict[str, Any]]: List of raw tool schemas in function-calling format.
        """
        formatted: list[dict[str, t.Any]] = []
        raw_tools = self._composio.tools.get_raw_composio_tools(
            tools=tool_slugs, search=search, toolkits=toolkits, scopes=scopes, limit=limit
        )

        for tool in raw_tools:
            try:
                name = tool.slug  # type: ignore[attr-defined]
                description = getattr(tool, "description", "") or "Composio tool"
                params = getattr(tool, "input_parameters", None)
                if not params:
                    # Minimal shape if schema missing
                    params = {"type": "object", "properties": {}}
                formatted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": params,
                        },
                    }
                )
            except Exception as e:
                logger.warning("Failed to map Composio tool schema: %s", e)
                continue
        return formatted

    def execute(
        self,
        *,
        slug: str,
        arguments: dict[str, t.Any],
        user_id: str | None = None,
        connected_account_id: str | None = None,
        custom_auth_params: dict[str, t.Any] | None = None,
        custom_connection_data: dict[str, t.Any] | None = None,
        text: str | None = None,
        version: str | None = None,
        toolkit_versions: t.Any | None = None,
        modifiers: t.Any | None = None,
    ) -> dict[str, t.Any]:
        """
        Execute a Composio tool and return a normalized response dict.

        Args:
            slug (str): Tool slug to execute.
            arguments (dict[str, Any]): Arguments for the tool.
            user_id (str | None): Optional user ID.
            connected_account_id (str | None): Optional connected account ID.
            custom_auth_params (dict[str, Any] | None): Optional custom auth params.
            custom_connection_data (dict[str, Any] | None): Optional custom connection data.
            text (str | None): Optional text input.
            version (str | None): Optional version.
            toolkit_versions (Any | None): Optional toolkit versions.
            modifiers (Any | None): Optional modifiers.

        Returns:
            dict[str, Any]: Normalized response dict with keys: successful, data, error.
        """
        resp = self._composio.tools.execute(
            slug=slug,
            arguments=arguments,
            user_id=user_id,
            connected_account_id=connected_account_id,
            custom_auth_params=custom_auth_params,
            custom_connection_data=custom_connection_data,
            text=text,
            version=version,
            toolkit_versions=toolkit_versions,
            modifiers=modifiers,
        )

        # The SDK returns a TypedDict-like object; ensure plain dict
        if hasattr(resp, "copy") and not isinstance(resp, dict):  # e.g., TypedDict proxy
            try:
                resp = dict(resp)  # type: ignore[assignment]
            except Exception as exc:
                logger.debug("Could not coerce Composio response to dict: %s", exc)

        # Normalize key presence
        successful = bool(resp.get("successful", False))  # type: ignore[arg-type]
        data = resp.get("data")
        error = resp.get("error")
        return {"successful": successful, "data": data, "error": error}
