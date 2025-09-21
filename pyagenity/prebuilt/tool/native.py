"""Native essential tools (no external deps).

Tools included:
- Files: read_file, write_file, list_dir (sandboxed)
- HTTP: http_get (safe, blocks localhost/private ranges, size/time limits)
- Python: python_eval (restricted eval/exec)

Each tool is a plain callable and marked with metadata attributes that ToolNode
surfaces in tool schemas: ``_py_tool_tags``, ``_py_tool_provider``, and
``_py_tool_capabilities``.
"""

from __future__ import annotations

import ast
import ipaddress
import json
import math
import socket
import time
import typing as t
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pyagenity.graph.tool_node import ToolNode


Provider = "native"


def _attach_metadata(
    fn: t.Callable,
    *,
    tags: list[str] | None = None,
    capabilities: list[str] | None = None,
) -> t.Callable:
    # Attach discoverable metadata for ToolNode schema enrichment
    fn._py_tool_tags = tags or []  # type: ignore[attr-defined]
    fn._py_tool_provider = Provider  # type: ignore[attr-defined]
    fn._py_tool_capabilities = capabilities or []  # type: ignore[attr-defined]
    return fn


# ---------- Files (sandboxed) ----------


def _ensure_in_sandbox(root: Path, p: Path) -> Path:
    p = (root / p).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise PermissionError("Path escapes sandbox")
    return p


def _make_file_tools(sandbox_root: str | None = None) -> list[t.Callable]:
    root = Path(sandbox_root or ".sandbox").resolve()
    root.mkdir(parents=True, exist_ok=True)

    def read_file(path: str, max_bytes: int = 65536, encoding: str = "utf-8") -> str:
        """Read a text file from sandbox.

        Args:
            path: Relative path inside sandbox
            max_bytes: Max bytes to read (to avoid huge files)
            encoding: Text encoding
        """
        p = _ensure_in_sandbox(root, Path(path))
        data = Path(p).read_bytes()[:max_bytes]
        return data.decode(encoding, errors="replace")

    def write_file(
        path: str,
        content: str,
        overwrite: bool = False,
        encoding: str = "utf-8",
    ) -> str:
        """Write a text file into sandbox.

        Args:
            path: Relative path inside sandbox
            content: Text content
            overwrite: Allow overwriting existing files
            encoding: Text encoding
        """
        p = _ensure_in_sandbox(root, Path(path))
        if p.exists() and not overwrite:
            raise FileExistsError("File exists; set overwrite=True to replace")
        p.parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(content, encoding=encoding)
        return str(p.relative_to(root))

    def list_dir(path: str = ".", max_entries: int = 100) -> list[str]:
        """List entries under a sandboxed directory (non-recursive)."""
        p = _ensure_in_sandbox(root, Path(path))
        if not p.exists() or not p.is_dir():
            return []
        entries = []
        for child in p.iterdir():
            entries.append(child.name + ("/" if child.is_dir() else ""))
            if len(entries) >= max_entries:
                break
        return entries

    return [
        _attach_metadata(read_file, tags=["io", "file", "read"], capabilities=["files"]),
        _attach_metadata(
            write_file,
            tags=["io", "file", "write"],
            capabilities=["files"],
        ),
        _attach_metadata(list_dir, tags=["io", "file", "list"], capabilities=["files"]),
    ]


# ---------- HTTP (safe GET) ----------


def _is_private_host(host: str) -> bool:
    try:
        # Resolve and check each address
        for res in socket.getaddrinfo(host, None):
            addr = res[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return True
    except Exception:
        return True
    return False


def http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 15,
    max_bytes: int = 200_000,
    allow_localhost: bool = False,
) -> dict:
    """Fetch a URL with safety checks.

    - Only http/https schemes
    - Blocks localhost/private ranges unless explicitly allowed
    - Limits response size and time
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are allowed")
    host = parsed.hostname or ""
    if not allow_localhost and _is_private_host(host):
        raise PermissionError("Access to local/private addresses is blocked")

    # Guard again to satisfy security linters and bandit
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    # At this point, scheme is asserted to be http/https only
    req = Request(url, headers=headers or {"User-Agent": "pyagenity-native/1.0"})  # nosec B310  # noqa: S310
    start = time.time()
    with urlopen(req, timeout=timeout) as resp:  # nosec B310  # noqa: S310
        status = resp.getcode()
        ctype = resp.headers.get("Content-Type", "application/octet-stream")
        data = resp.read(max_bytes)
        duration_ms = int((time.time() - start) * 1000)
        try:
            text = data.decode("utf-8") if "text" in ctype or "json" in ctype else None
        except Exception:
            text = None
        return {
            "status": status,
            "content_type": ctype,
            "bytes": len(data),
            "elapsed_ms": duration_ms,
            "text": text,
        }


_attach_metadata(http_get, tags=["net", "http", "get"], capabilities=["network"])


# ---------- Python (restricted eval/exec) ----------


SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
}

SAFE_GLOBALS = {"__builtins__": SAFE_BUILTINS, "math": math}


def python_eval(code: str, mode: str = "eval") -> str:
    """Evaluate small Python snippets safely.

    - mode="eval" for expressions, mode="exec" for small blocks
    - No import statements, attribute access, or dunder names allowed
    """
    if "__" in code:
        raise PermissionError("Dunder access is not allowed")
    is_exec = mode == "exec"
    tree = ast.parse(code, mode="exec" if is_exec else "eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import | ast.ImportFrom | ast.Attribute)):
            raise PermissionError("Imports/attributes are not allowed")
    if is_exec:
        compiled = compile(code, "<pyagenity>", "exec")
        try:
            exec(compiled, SAFE_GLOBALS, {})  # nosec  # noqa: S102
            return "ok"
        except Exception as e:
            return f"Error: {e}"
    compiled = compile(code, "<pyagenity>", "eval")
    try:
        result = eval(compiled, SAFE_GLOBALS, {})  # nosec  # noqa: S307
        try:
            return json.dumps(result)
        except Exception:
            return str(result)
    except Exception as e:
        return f"Error: {e}"


_attach_metadata(python_eval, tags=["analysis", "python"], capabilities=["compute"])


def get_native_tools(sandbox_root: str | None = None) -> list[t.Callable]:
    """Return the default set of native tools, optionally binding a sandbox root."""
    tools: list[t.Callable] = []
    tools.extend(_make_file_tools(sandbox_root))
    tools.extend([http_get, python_eval])
    return tools


def filter_tools(
    tools: list[t.Callable], *, tags: list[str] | None = None, capabilities: list[str] | None = None
) -> list[t.Callable]:
    """Filter tools by metadata tags/capabilities.

    Args:
        tools: A list of tool callables returned by get_native_tools or custom.
        tags: If provided, keep tools that include ALL of these tags.
        capabilities: If provided, keep tools that include ALL of these capabilities.
    """
    out: list[t.Callable] = []
    need_tags = set(tags or [])
    need_caps = set(capabilities or [])
    for fn in tools:
        have_tags = set(getattr(fn, "_py_tool_tags", []) or [])
        have_caps = set(getattr(fn, "_py_tool_capabilities", []) or [])
        if need_tags and not need_tags.issubset(have_tags):
            continue
        if need_caps and not need_caps.issubset(have_caps):
            continue
        out.append(fn)
    return out


def build_native_tool_node(
    sandbox_root: str | None = None,
    *,
    include_tags: list[str] | None = None,
    include_capabilities: list[str] | None = None,
    composio_adapter: t.Any | None = None,
    langchain_adapter: t.Any | None = None,
):
    """Convenience builder that returns a ToolNode preloaded with native tools.

    This helps wire native essentials into prebuilt agents quickly.
    """

    tools = get_native_tools(sandbox_root)
    if include_tags or include_capabilities:
        tools = filter_tools(tools, tags=include_tags, capabilities=include_capabilities)
    return ToolNode(
        tools,
        composio_adapter=composio_adapter,
        langchain_adapter=langchain_adapter,
    )
