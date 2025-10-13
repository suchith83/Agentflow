"""Tests for LangChain adapter functionality."""
from unittest.mock import Mock, patch

import pytest

from agentflow.adapters.tools.langchain_adapter import LangChainAdapter, LangChainToolWrapper


class MockLangChainTool:
    """Mock LangChain tool for testing."""
    
    def __init__(self, name="mock_tool", description="Mock tool for testing"):
        self.name = name
        self.description = description
        self.args = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["query"]
        }
    
    def invoke(self, arguments):
        return f"Mock result for {arguments}"
    
    def run(self, arguments):
        return f"Mock run result for {arguments}"


class MockStructuredTool:
    """Mock StructuredTool with func attribute."""
    
    def __init__(self, name="structured_tool"):
        self.name = name
        self.description = "Mock structured tool"
        self.func = lambda query: f"Structured result: {query}"
        self.args_schema = MockArgsSchema


class MockArgsSchema:
    """Mock pydantic args schema."""
    
    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {
                "input_text": {"type": "string"},
                "max_length": {"type": "integer"}
            },
            "required": ["input_text"]
        }


class MockPydanticV1Schema:
    """Mock pydantic v1 args schema."""
    
    @classmethod
    def schema(cls):
        return {
            "type": "object",
            "properties": {
                "search_term": {"type": "string"}
            },
            "required": ["search_term"]
        }


class MockCallableTool:
    """Mock callable tool with _run method."""
    
    def __init__(self):
        self.name = "callable_tool"
        self.description = "Callable tool"
    
    def _run(self, arguments):
        """LangChain-style _run method that accepts a dict argument."""
        query = arguments.get("query", "")
        count = arguments.get("count", 5)
        return f"Callable result: {query} (count: {count})"


class TestLangChainToolWrapper:
    """Test class for LangChainToolWrapper."""
    
    def test_wrapper_init_basic(self):
        """Test basic wrapper initialization."""
        tool = MockLangChainTool()
        wrapper = LangChainToolWrapper(tool)
        
        assert wrapper.name == "mock_tool"
        assert wrapper.description == "Mock tool for testing"
        assert wrapper._tool is tool
    
    def test_wrapper_init_with_overrides(self):
        """Test wrapper initialization with name/description overrides."""
        tool = MockLangChainTool()
        wrapper = LangChainToolWrapper(
            tool,
            name="custom_name",
            description="Custom description"
        )
        
        assert wrapper.name == "custom_name"
        assert wrapper.description == "Custom description"
    
    def test_wrapper_init_no_name_attribute(self):
        """Test wrapper initialization when tool has no name attribute."""
        tool = Mock()
        del tool.name  # Remove name attribute
        wrapper = LangChainToolWrapper(tool)
        
        # Should use class name in snake_case
        assert wrapper.name == "mock"
    
    def test_default_name_conversion(self):
        """Test _default_name converts class names to snake_case."""
        class MyTestTool:
            pass
        
        class HTTPClient:
            pass
        
        tool1 = MyTestTool()
        tool2 = HTTPClient()
        
        assert LangChainToolWrapper._default_name(tool1) == "my_test_tool"
        assert LangChainToolWrapper._default_name(tool2) == "h_t_t_p_client"
    
    def test_resolve_callable_structured_tool(self):
        """Test _resolve_callable with StructuredTool.func."""
        tool = MockStructuredTool()
        callable_func = LangChainToolWrapper._resolve_callable(tool)
        
        assert callable_func is tool.func
    
    def test_resolve_callable_with_run_method(self):
        """Test _resolve_callable with run method."""
        tool = Mock(spec=['run'])  # Only has run method, no func/coroutine
        tool.run = Mock()
        callable_func = LangChainToolWrapper._resolve_callable(tool)
        assert callable_func is tool.run
    
    def test_resolve_callable_with_private_run(self):
        """Test _resolve_callable with _run method."""
        tool = MockCallableTool()
        callable_func = LangChainToolWrapper._resolve_callable(tool)

        # Both should be the _run method
        assert callable_func == tool._run

    def test_resolve_callable_none(self):
        """Test _resolve_callable returns None when no callable found."""
        tool = Mock()
        del tool.func
        del tool.coroutine
        del tool.run
        del tool._run
        
        callable_func = LangChainToolWrapper._resolve_callable(tool)
        assert callable_func is None
    
    def test_json_schema_from_args(self):
        """Test _json_schema_from_args_schema with args attribute."""
        tool = MockLangChainTool()
        wrapper = LangChainToolWrapper(tool)
        
        schema = wrapper._json_schema_from_args_schema()
        
        assert schema == {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["query"]
        }
    
    def test_json_schema_from_pydantic_v2(self):
        """Test _json_schema_from_args_schema with pydantic v2 args_schema."""
        tool = MockStructuredTool()
        wrapper = LangChainToolWrapper(tool)
        
        schema = wrapper._json_schema_from_args_schema()
        
        expected = {
            "type": "object",
            "properties": {
                "input_text": {"type": "string"},
                "max_length": {"type": "integer"}
            },
            "required": ["input_text"]
        }
        assert schema == expected
    
    def test_json_schema_from_pydantic_v1(self):
        """Test _json_schema_from_args_schema with pydantic v1 args_schema."""
        tool = Mock()
        tool.args = None  # No args attribute
        tool.args_schema = MockPydanticV1Schema
        wrapper = LangChainToolWrapper(tool)
        
        schema = wrapper._json_schema_from_args_schema()
        
        expected = {
            "type": "object",
            "properties": {
                "search_term": {"type": "string"}
            },
            "required": ["search_term"]
        }
        assert schema == expected
    
    def test_json_schema_none(self):
        """Test _json_schema_from_args_schema returns None when no schema."""
        tool = Mock()
        del tool.args
        del tool.args_schema
        wrapper = LangChainToolWrapper(tool)
        
        schema = wrapper._json_schema_from_args_schema()
        assert schema is None
    
    def test_infer_schema_from_signature(self):
        """Test _infer_schema_from_signature with callable function."""
        tool = MockCallableTool()
        wrapper = LangChainToolWrapper(tool)
        
        schema = wrapper._infer_schema_from_signature()
        
        # The _run method now has signature (arguments) so expect that in the schema
        expected = {
            "type": "object", 
            "properties": {
                "arguments": {}
            },
            "required": ["arguments"]
        }
        assert schema == expected

    def test_infer_schema_empty_fallback(self):
        """Test _infer_schema_from_signature fallback to empty schema."""
        tool = Mock()
        del tool.invoke  # No invoke method
        wrapper = LangChainToolWrapper(tool)
        wrapper._callable = None
        
        schema = wrapper._infer_schema_from_signature()
        
        assert schema == {"type": "object", "properties": {}}
    
    def test_map_annotation_to_json_type(self):
        """Test _map_annotation_to_json_type type mapping."""
        assert LangChainToolWrapper._map_annotation_to_json_type(str) == "string"
        assert LangChainToolWrapper._map_annotation_to_json_type(int) == "integer"
        assert LangChainToolWrapper._map_annotation_to_json_type(float) == "number"
        assert LangChainToolWrapper._map_annotation_to_json_type(bool) == "boolean"
        assert LangChainToolWrapper._map_annotation_to_json_type(list) == "array"
        assert LangChainToolWrapper._map_annotation_to_json_type(dict) == "object"
        assert LangChainToolWrapper._map_annotation_to_json_type(tuple) == "array"
        assert LangChainToolWrapper._map_annotation_to_json_type(set) == "array"
    
    def test_to_schema(self):
        """Test to_schema method."""
        tool = MockLangChainTool()
        wrapper = LangChainToolWrapper(tool)
        
        schema = wrapper.to_schema()
        
        expected = {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "Mock tool for testing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "count": {"type": "integer"}
                    },
                    "required": ["query"]
                }
            }
        }
        assert schema == expected
    
    def test_execute_with_invoke(self):
        """Test execute method using invoke."""
        tool = MockLangChainTool()
        wrapper = LangChainToolWrapper(tool)
        
        result = wrapper.execute({"query": "test"})
        
        assert result["successful"] is True
        assert result["data"] == "Mock result for {'query': 'test'}"
        assert result["error"] is None
    
    def test_execute_with_run(self):
        """Test execute method using run when invoke not available."""
        tool = Mock()
        tool.run = lambda args: f"Run result: {args}"
        del tool.invoke  # No invoke method
        wrapper = LangChainToolWrapper(tool)
        
        result = wrapper.execute({"input": "test"})
        
        assert result["successful"] is True
        assert result["data"] == "Run result: {'input': 'test'}"
    
    def test_execute_with_private_run(self):
        """Test execute method using _run."""
        tool = MockCallableTool()
        wrapper = LangChainToolWrapper(tool)
        
        # Mock the wrapper to test _run path by creating a tool without invoke/run
        mock_tool = Mock(spec=['_run'])  # Only has _run method
        mock_tool._run = tool._run
        wrapper._tool = mock_tool
        
        result = wrapper.execute({"query": "test", "count": 3})
        
        assert result["successful"] is True
        assert result["data"] == "Callable result: test (count: 3)"
    
    def test_execute_with_callable(self):
        """Test execute method using callable function."""
        def mock_func(query: str):
            return f"Func result: {query}"
        
        tool = Mock()
        del tool.invoke
        del tool.run
        del tool._run
        wrapper = LangChainToolWrapper(tool)
        wrapper._callable = mock_func
        
        result = wrapper.execute({"query": "test"})
        
        assert result["successful"] is True
        assert result["data"] == "Func result: test"
    
    def test_execute_no_method_error(self):
        """Test execute method when no execution method available."""
        tool = Mock()
        del tool.invoke
        del tool.run
        del tool._run
        wrapper = LangChainToolWrapper(tool)
        wrapper._callable = None
        
        result = wrapper.execute({"query": "test"})
        
        assert result["successful"] is False
        assert result["data"] is None
        assert "does not support invoke/run/_run/callable" in result["error"]
    
    def test_execute_exception_handling(self):
        """Test execute method handles exceptions properly."""
        tool = Mock()
        tool.invoke = Mock(side_effect=ValueError("Test error"))
        wrapper = LangChainToolWrapper(tool)
        
        result = wrapper.execute({"query": "test"})
        
        assert result["successful"] is False
        assert result["data"] is None
        assert result["error"] == "Test error"
    
    def test_execute_non_json_serializable_result(self):
        """Test execute handles non-JSON serializable results."""
        class CustomObject:
            def __str__(self):
                return "custom_object_string"
        
        tool = Mock()
        tool.invoke = Mock(return_value=CustomObject())
        wrapper = LangChainToolWrapper(tool)
        
        result = wrapper.execute({"query": "test"})
        
        assert result["successful"] is True
        assert result["data"] == "custom_object_string"


class TestLangChainAdapter:
    """Test class for LangChainAdapter."""
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', False)
    def test_adapter_init_no_langchain(self):
        """Test adapter initialization fails when LangChain not available."""
        with pytest.raises(ImportError, match="LangChainAdapter requires 'langchain-core'"):
            LangChainAdapter()
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_adapter_init_success(self):
        """Test successful adapter initialization."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        assert adapter._registry == {}
        assert adapter._autoload is False
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_adapter_init_with_autoload(self):
        """Test adapter initialization with autoload enabled."""
        adapter = LangChainAdapter(autoload_default_tools=True)
        assert adapter._autoload is True
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_is_available_true(self):
        """Test is_available returns True when LangChain is available."""
        assert LangChainAdapter.is_available() is True
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', False)
    def test_is_available_false(self):
        """Test is_available returns False when LangChain not available."""
        assert LangChainAdapter.is_available() is False
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_register_tool(self):
        """Test register_tool method."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        tool = MockLangChainTool()
        
        name = adapter.register_tool(tool)
        
        assert name == "mock_tool"
        assert "mock_tool" in adapter._registry
        assert adapter._registry["mock_tool"]._tool is tool
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_register_tool_with_overrides(self):
        """Test register_tool with name and description overrides."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        tool = MockLangChainTool()
        
        name = adapter.register_tool(
            tool,
            name="custom_tool",
            description="Custom description"
        )
        
        assert name == "custom_tool"
        assert adapter._registry["custom_tool"].name == "custom_tool"
        assert adapter._registry["custom_tool"].description == "Custom description"
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_register_tools_multiple(self):
        """Test register_tools method with multiple tools."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        tool1 = MockLangChainTool(name="tool1")
        tool2 = MockLangChainTool(name="tool2")
        
        names = adapter.register_tools([tool1, tool2])
        
        assert names == ["tool1", "tool2"]
        assert len(adapter._registry) == 2
        assert "tool1" in adapter._registry
        assert "tool2" in adapter._registry
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_list_tools_for_llm_empty(self):
        """Test list_tools_for_llm with empty registry."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        
        tools = adapter.list_tools_for_llm()
        
        assert tools == []
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_list_tools_for_llm_with_tools(self):
        """Test list_tools_for_llm with registered tools."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        tool = MockLangChainTool()
        adapter.register_tool(tool)
        
        tools = adapter.list_tools_for_llm()
        
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "mock_tool"
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_execute_success(self):
        """Test execute method with successful tool execution."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        tool = MockLangChainTool()
        adapter.register_tool(tool)
        
        result = adapter.execute(name="mock_tool", arguments={"query": "test"})
        
        assert result["successful"] is True
        assert "Mock result for" in result["data"]
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_execute_tool_not_found(self):
        """Test execute method with unknown tool name."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        
        result = adapter.execute(name="unknown_tool", arguments={})
        
        assert result["successful"] is False
        assert result["data"] is None
        assert result["error"] == "Unknown LangChain tool: unknown_tool"
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_list_tools_autoload_triggered(self):
        """Test list_tools_for_llm triggers autoload when enabled."""
        adapter = LangChainAdapter(autoload_default_tools=True)
        
        with patch.object(adapter, '_try_autoload_defaults') as mock_autoload:
            adapter.list_tools_for_llm()
            mock_autoload.assert_called_once()
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_execute_autoload_triggered(self):
        """Test execute triggers autoload when enabled and tool not found."""
        adapter = LangChainAdapter(autoload_default_tools=True)
        
        with patch.object(adapter, '_try_autoload_defaults') as mock_autoload:
            adapter.execute(name="unknown_tool", arguments={})
            mock_autoload.assert_called_once()


class TestLangChainAdapterAutoload:
    """Test class for LangChain adapter autoload functionality."""
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_create_tavily_search_tool_success(self):
        """Test _create_tavily_search_tool with successful import."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.TavilySearch.return_value = Mock()
            mock_import.return_value = mock_module
            
            tool = adapter._create_tavily_search_tool()
            
            assert tool is not None
            mock_import.assert_called_with('langchain_tavily')
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_create_tavily_search_tool_fallback(self):
        """Test _create_tavily_search_tool fallback to community tool."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        
        with patch('importlib.import_module') as mock_import:
            # First import fails (langchain_tavily), second succeeds (community)
            def side_effect(module):
                if module == 'langchain_tavily':
                    raise ImportError("Module not found")
                elif module == 'langchain_community.tools.tavily_search':
                    mock_module = Mock()
                    mock_module.TavilySearchResults.return_value = Mock()
                    return mock_module
                else:
                    raise ImportError("Unexpected module")
            
            mock_import.side_effect = side_effect
            
            tool = adapter._create_tavily_search_tool()
            
            assert tool is not None
            assert mock_import.call_count == 2
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_create_tavily_search_tool_failure(self):
        """Test _create_tavily_search_tool with import failure."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        
        with patch('importlib.import_module', side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="Tavily tool requires"):
                adapter._create_tavily_search_tool()
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_create_requests_get_tool_success(self):
        """Test _create_requests_get_tool with successful import."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        
        with patch('importlib.import_module') as mock_import:
            mock_tool_module = Mock()
            mock_util_module = Mock()
            mock_wrapper = Mock()
            mock_util_module.TextRequestsWrapper.return_value = mock_wrapper
            mock_tool_module.RequestsGetTool.return_value = Mock()
            
            def side_effect(module):
                if 'tool' in module:
                    return mock_tool_module
                else:
                    return mock_util_module
            
            mock_import.side_effect = side_effect
            
            tool = adapter._create_requests_get_tool()
            
            assert tool is not None
            assert mock_import.call_count == 2
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_create_requests_get_tool_failure(self):
        """Test _create_requests_get_tool with import failure."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        
        with patch('importlib.import_module', side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="Requests tool requires"):
                adapter._create_requests_get_tool()
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_try_autoload_defaults_success(self):
        """Test _try_autoload_defaults with successful tool creation."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        
        with patch.object(adapter, '_create_tavily_search_tool') as mock_tavily:
            with patch.object(adapter, '_create_requests_get_tool') as mock_requests:
                mock_tavily.return_value = MockLangChainTool("tavily")
                mock_requests.return_value = MockLangChainTool("requests")
                
                adapter._try_autoload_defaults()
                
                assert len(adapter._registry) == 2
                assert "tavily_search" in adapter._registry
                assert "requests_get" in adapter._registry
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_try_autoload_defaults_partial_failure(self):
        """Test _try_autoload_defaults with partial failure."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        
        with patch.object(adapter, '_create_tavily_search_tool') as mock_tavily:
            with patch.object(adapter, '_create_requests_get_tool') as mock_requests:
                mock_tavily.side_effect = ImportError("Tavily failed")
                mock_requests.return_value = MockLangChainTool("requests")
                
                # Should not raise, just log and continue
                adapter._try_autoload_defaults()
                
                assert len(adapter._registry) == 1
                assert "requests_get" in adapter._registry
                assert "tavily_search" not in adapter._registry
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_try_autoload_defaults_complete_failure(self):
        """Test _try_autoload_defaults with complete failure."""
        adapter = LangChainAdapter(autoload_default_tools=False)
        
        with patch.object(adapter, '_create_tavily_search_tool') as mock_tavily:
            with patch.object(adapter, '_create_requests_get_tool') as mock_requests:
                mock_tavily.side_effect = ImportError("Tavily failed")
                mock_requests.side_effect = ImportError("Requests failed")
                
                # Should not raise, just log and continue
                adapter._try_autoload_defaults()
                
                assert len(adapter._registry) == 0


class TestLangChainAdapterEdgeCases:
    """Test class for LangChain adapter edge cases."""
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_wrapper_with_coroutine_attribute(self):
        """Test wrapper with tool that has coroutine attribute."""
        tool = Mock()
        tool.coroutine = lambda x: f"Coroutine result: {x}"
        del tool.func  # No func attribute
        wrapper = LangChainToolWrapper(tool)
        
        assert wrapper._callable is tool.coroutine
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_wrapper_exception_in_callable_resolution(self):
        """Test wrapper handles exceptions during callable resolution."""
        tool = Mock()
        # Simulate exception when accessing func
        type(tool).func = property(lambda self: exec('raise ValueError("test error")'))
        wrapper = LangChainToolWrapper(tool)
        
        # Should not crash, should fallback to other methods
        assert wrapper is not None
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_wrapper_pydantic_schema_exception(self):
        """Test wrapper handles pydantic schema exceptions."""
        tool = Mock()
        tool.args = None
        tool.args_schema = Mock()
        tool.args_schema.model_json_schema.side_effect = ValueError("Schema error")
        wrapper = LangChainToolWrapper(tool)
        
        schema = wrapper._json_schema_from_args_schema()
        assert schema is None
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_map_annotation_exception_handling(self):
        """Test _map_annotation_to_json_type handles exceptions."""
        # Test with an object that will cause get_origin to fail
        class ProblematicType:
            pass
        
        result = LangChainToolWrapper._map_annotation_to_json_type(ProblematicType())
        assert result is None
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_infer_schema_signature_exception(self):
        """Test _infer_schema_from_signature handles signature exceptions."""
        def problematic_func():
            pass
        
        # Mock a tool where inspect.signature would fail
        tool = Mock()
        tool.invoke = problematic_func
        wrapper = LangChainToolWrapper(tool)
        
        with patch('inspect.signature', side_effect=ValueError("Signature error")):
            schema = wrapper._infer_schema_from_signature()
            assert schema == {"type": "object", "properties": {}}
    
    @patch('agentflow.adapters.tools.langchain_adapter.HAS_LANGCHAIN', True)
    def test_execute_json_dumps_check(self):
        """Test execute method JSON serialization check."""
        class JsonSerializable:
            def __init__(self):
                self.data = "test"
        
        class NonJsonSerializable:
            def __init__(self):
                self.circular_ref = self
        
        tool1 = Mock()
        tool1.invoke = Mock(return_value=JsonSerializable())
        wrapper1 = LangChainToolWrapper(tool1)
        
        result1 = wrapper1.execute({})
        assert result1["successful"] is True
        # The object should be converted to string since it's not a basic JSON type
        assert isinstance(result1["data"], str)
        
        tool2 = Mock()
        tool2.invoke = Mock(return_value=NonJsonSerializable())
        wrapper2 = LangChainToolWrapper(tool2)
        
        result2 = wrapper2.execute({})
        assert result2["successful"] is True
        assert isinstance(result2["data"], str)  # Converted to string