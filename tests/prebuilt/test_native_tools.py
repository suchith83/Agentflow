import json
from collections.abc import Callable

import pytest

from pyagenity.prebuilt.tool import build_native_tool_node, filter_tools, get_native_tools


def _tools_by_name(tools: list[Callable]) -> dict[str, Callable]:
    return {fn.__name__: fn for fn in tools}


def test_file_sandbox_write_read_list(tmp_path):
    tools = get_native_tools(sandbox_root=str(tmp_path))
    t = _tools_by_name(tools)

    # write, then read
    rel = t["write_file"]("folder/hello.txt", content="hi", overwrite=False)
    assert rel == "folder/hello.txt"
    text = t["read_file"]("folder/hello.txt")
    assert text == "hi"
    ls = t["list_dir"]("folder")
    assert ls == ["hello.txt"]


def test_file_sandbox_prevent_escape(tmp_path):
    tools = get_native_tools(sandbox_root=str(tmp_path))
    t = _tools_by_name(tools)

    with pytest.raises(PermissionError):
        t["write_file"]("../outside.txt", content="nope", overwrite=True)


def test_http_get_blocks_private():
    tools = get_native_tools()
    t = _tools_by_name(tools)
    with pytest.raises(PermissionError):
        t["http_get"]("http://127.0.0.1:8080")


def test_http_get_scheme_enforced():
    tools = get_native_tools()
    t = _tools_by_name(tools)
    with pytest.raises(ValueError):
        t["http_get"]("file:///etc/hosts")


def test_python_eval_safety_and_success():
    tools = get_native_tools()
    t = _tools_by_name(tools)

    # safe expr
    out = t["python_eval"]("1+2", mode="eval")
    assert json.loads(out) == 3

    # block imports and dunder
    with pytest.raises(PermissionError):
        t["python_eval"]("import os; os.getcwd()", mode="exec")
    with pytest.raises(PermissionError):
        t["python_eval"]("__import__('os')")


def test_filter_tools_and_build_node(tmp_path):
    tools = get_native_tools(sandbox_root=str(tmp_path))
    http_only = filter_tools(tools, tags=["http"])  # should pick http_get
    names = {f.__name__ for f in http_only}
    assert names == {"http_get"}

    # Build ToolNode and introspect metadata
    node = build_native_tool_node(sandbox_root=str(tmp_path))
    specs = node.get_local_tool()
    # Ensure x-pyagenity metadata present and provider is native
    assert any("x-pyagenity" in s and s["x-pyagenity"].get("provider") == "native" for s in specs)
