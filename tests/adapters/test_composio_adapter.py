"""
Comprehensive tests for the ComposioAdapter class.

This module tests the ComposioAdapter's functionality including initialization,
tool listing, and execution capabilities, with comprehensive mocking of the
Composio SDK to test all code paths.
"""

import logging
from unittest.mock import Mock, MagicMock, patch
import pytest

from agentflow.adapters.tools.composio_adapter import ComposioAdapter


class MockComposioSDK:
    """Mock Composio SDK for testing."""
    
    def __init__(self, api_key=None, provider=None, file_download_dir=None, toolkit_versions=None):
        self.api_key = api_key
        self.provider = provider
        self.file_download_dir = file_download_dir
        self.toolkit_versions = toolkit_versions
        self.tools = MockComposioTools()


class MockComposioTools:
    """Mock Composio tools interface."""
    
    def __init__(self):
        self._mock_tools = [
            {
                "type": "function",
                "function": {
                    "name": "GITHUB_LIST_STARGAZERS",
                    "description": "List stargazers for a repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string"},
                            "repo": {"type": "string"}
                        },
                        "required": ["owner", "repo"]
                    }
                }
            }
        ]
        self._raw_tools: list = [MockRawTool()]
        self._execution_result: dict | object = {
            "successful": True,
            "data": {"stargazers": ["user1", "user2"]},
            "error": None
        }
    
    def get(self, user_id=None, tools=None, toolkits=None, search=None, scopes=None, limit=None):
        """Mock the tools.get method."""
        return self._mock_tools
    
    def get_raw_composio_tools(self, tools=None, search=None, toolkits=None, scopes=None, limit=None):
        """Mock the get_raw_composio_tools method."""
        return self._raw_tools
    
    def execute(self, slug, arguments, user_id=None, connected_account_id=None, **kwargs):
        """Mock the execute method."""
        return self._execution_result


class MockRawTool:
    """Mock raw Composio tool."""
    
    def __init__(self):
        self.slug = "RAW_TOOL_EXAMPLE"
        self.description = "Example raw tool"
        self.input_parameters = {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        }


class TestComposioAdapterAvailability:
    """Test ComposioAdapter availability checking."""
    
    def test_is_available_when_composio_imported(self):
        """Test is_available returns True when composio is available."""
        with patch('agentflow.adapters.tools.composio_adapter.HAS_COMPOSIO', True):
            assert ComposioAdapter.is_available() is True
    
    def test_is_available_when_composio_not_imported(self):
        """Test is_available returns False when composio is not available."""
        with patch('agentflow.adapters.tools.composio_adapter.HAS_COMPOSIO', False):
            assert ComposioAdapter.is_available() is False


class TestComposioAdapterInitialization:
    """Test ComposioAdapter initialization."""
    
    @patch('agentflow.adapters.tools.composio_adapter.HAS_COMPOSIO', True)
    @patch('agentflow.adapters.tools.composio_adapter.Composio')
    def test_init_with_default_params(self, mock_composio_class):
        """Test initialization with default parameters."""
        mock_composio_instance = Mock()
        mock_composio_class.return_value = mock_composio_instance
        
        adapter = ComposioAdapter(
            api_key="",
            provider=None,
            file_download_dir="",
            toolkit_versions=None
        )
        
        assert adapter._composio == mock_composio_instance
        mock_composio_class.assert_called_once_with(
            api_key=None,
            provider=None,
            file_download_dir=None,
            toolkit_versions=None
        )
    
    @patch('agentflow.adapters.tools.composio_adapter.HAS_COMPOSIO', True)
    @patch('agentflow.adapters.tools.composio_adapter.Composio')
    def test_init_with_custom_params(self, mock_composio_class):
        """Test initialization with custom parameters."""
        mock_composio_instance = Mock()
        mock_composio_class.return_value = mock_composio_instance
        
        adapter = ComposioAdapter(
            api_key="test-key",
            provider="test-provider",
            file_download_dir="/test/dir",
            toolkit_versions={"toolkit1": "v1.0"}
        )
        
        assert adapter._composio == mock_composio_instance
        mock_composio_class.assert_called_once_with(
            api_key="test-key",
            provider="test-provider",
            file_download_dir="/test/dir",
            toolkit_versions={"toolkit1": "v1.0"}
        )
    
    @patch('agentflow.adapters.tools.composio_adapter.HAS_COMPOSIO', False)
    def test_init_raises_import_error_when_composio_unavailable(self):
        """Test initialization raises ImportError when composio is not available."""
        with pytest.raises(ImportError) as exc_info:
            ComposioAdapter(
                api_key="test-key",
                provider="test-provider",
                file_download_dir="/test/dir",
                toolkit_versions=None
            )
        
        assert "ComposioAdapter requires 'composio' package" in str(exc_info.value)
        assert "pip install 10xscale-agentflow[composio]" in str(exc_info.value)


class TestComposioAdapterToolListing:
    """Test ComposioAdapter tool listing methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_composio = MockComposioSDK()
        
        # Patch the availability and Composio class
        self.composio_patcher = patch('agentflow.adapters.tools.composio_adapter.HAS_COMPOSIO', True)
        self.class_patcher = patch('agentflow.adapters.tools.composio_adapter.Composio', return_value=self.mock_composio)
        
        self.composio_patcher.start()
        self.class_patcher.start()
        
        self.adapter = ComposioAdapter(
            api_key="",
            provider=None,
            file_download_dir="",
            toolkit_versions=None
        )
    
    def teardown_method(self):
        """Clean up patches."""
        self.composio_patcher.stop()
        self.class_patcher.stop()
    
    def test_list_tools_for_llm_basic(self):
        """Test basic tool listing for LLM."""
        tools = self.adapter.list_tools_for_llm(user_id="test-user")
        
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "GITHUB_LIST_STARGAZERS"
        assert tools[0]["function"]["description"] == "List stargazers for a repository"
        assert "parameters" in tools[0]["function"]
    
    def test_list_tools_for_llm_with_parameters(self):
        """Test tool listing with various parameters."""
        tools = self.adapter.list_tools_for_llm(
            user_id="test-user",
            tool_slugs=["GITHUB_LIST_STARGAZERS"],
            toolkits=["github"],
            search="stargazers",
            scopes=["repo"],
            limit=10
        )
        
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "GITHUB_LIST_STARGAZERS"
    
    def test_list_tools_for_llm_empty_result(self):
        """Test tool listing when no tools are returned."""
        self.mock_composio.tools._mock_tools = []
        self.mock_composio.tools._raw_tools = []
        
        tools = self.adapter.list_tools_for_llm(user_id="test-user")
        
        assert len(tools) == 0
    
    def test_list_tools_for_llm_non_conforming_tools(self):
        """Test handling of non-conforming tool objects."""
        # Mock tools that don't follow expected structure
        self.mock_composio.tools._mock_tools = [
            {"invalid": "structure"},
            {"type": "function"},  # Missing function key
            {"type": "function", "function": {"name": "test", "parameters": {"type": "object"}}},  # Valid one
        ]
        
        with patch('agentflow.adapters.tools.composio_adapter.logger') as mock_logger:
            tools = self.adapter.list_tools_for_llm(user_id="test-user")
            
            # Should include at least the valid tool
            assert len(tools) >= 1
            # The valid tool should be included
            valid_tool_names = [tool["function"]["name"] for tool in tools]
            assert "test" in valid_tool_names
    
    def test_list_tools_for_llm_fallback_to_raw(self):
        """Test fallback to raw tools when formatted tools fail."""
        # Mock empty formatted tools but available raw tools
        self.mock_composio.tools._mock_tools = []
        
        tools = self.adapter.list_tools_for_llm(user_id="test-user")
        
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "RAW_TOOL_EXAMPLE"
    
    def test_list_raw_tools_for_llm_basic(self):
        """Test basic raw tool listing."""
        tools = self.adapter.list_raw_tools_for_llm()
        
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "RAW_TOOL_EXAMPLE"
        assert tools[0]["function"]["description"] == "Example raw tool"
        assert "parameters" in tools[0]["function"]
    
    def test_list_raw_tools_for_llm_with_parameters(self):
        """Test raw tool listing with parameters."""
        tools = self.adapter.list_raw_tools_for_llm(
            tool_slugs=["RAW_TOOL_EXAMPLE"],
            toolkits=["test"],
            search="example",
            scopes=["read"],
            limit=5
        )
        
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "RAW_TOOL_EXAMPLE"
    
    def test_list_raw_tools_for_llm_missing_attributes(self):
        """Test handling of raw tools with missing attributes."""
        # Create a simple tool with minimal attributes to avoid recursion issues
        class SimpleTool:
            def __init__(self):
                self.slug = "INCOMPLETE_TOOL"
                # Don't set description or input_parameters attributes
        
        incomplete_tool = SimpleTool()
        self.mock_composio.tools._raw_tools = [incomplete_tool]
        
        tools = self.adapter.list_raw_tools_for_llm()
        
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "INCOMPLETE_TOOL"
        assert tools[0]["function"]["description"] == "Composio tool"  # Default description
        assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {}}  # Default params
    
    def test_list_raw_tools_for_llm_exception_handling(self):
        """Test exception handling in raw tool processing."""
        # Create a tool that will cause an exception in the try block
        class FaultyTool:
            @property
            def slug(self):
                raise Exception("Test exception")
        
        faulty_tool = FaultyTool()
        self.mock_composio.tools._raw_tools = [faulty_tool, MockRawTool()]
        
        with patch('agentflow.adapters.tools.composio_adapter.logger') as mock_logger:
            tools = self.adapter.list_raw_tools_for_llm()
            
            # Should process the good tool and skip the faulty one
            good_tools = [tool for tool in tools if tool["function"]["name"] == "RAW_TOOL_EXAMPLE"]
            assert len(good_tools) == 1
            assert good_tools[0]["function"]["description"] == "Example raw tool"
            mock_logger.warning.assert_called()


class TestComposioAdapterExecution:
    """Test ComposioAdapter execution functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_composio = MockComposioSDK()
        
        # Patch the availability and Composio class
        self.composio_patcher = patch('agentflow.adapters.tools.composio_adapter.HAS_COMPOSIO', True)
        self.class_patcher = patch('agentflow.adapters.tools.composio_adapter.Composio', return_value=self.mock_composio)
        
        self.composio_patcher.start()
        self.class_patcher.start()

        self.adapter = ComposioAdapter(
            api_key="",
            provider=None,
            file_download_dir="",
            toolkit_versions=None
        )

    def teardown_method(self):
        """Clean up patches."""
        self.composio_patcher.stop()
        self.class_patcher.stop()
    
    def test_execute_basic(self):
        """Test basic tool execution."""
        result = self.adapter.execute(
            slug="GITHUB_LIST_STARGAZERS",
            arguments={"owner": "ComposioHQ", "repo": "composio"}
        )
        
        assert result["successful"] is True
        assert result["data"] == {"stargazers": ["user1", "user2"]}
        assert result["error"] is None
    
    def test_execute_with_all_parameters(self):
        """Test execution with all possible parameters."""
        result = self.adapter.execute(
            slug="GITHUB_LIST_STARGAZERS",
            arguments={"owner": "test", "repo": "test"},
            user_id="user-123",
            connected_account_id="account-456",
            custom_auth_params={"token": "secret"},
            custom_connection_data={"server": "api.github.com"},
            text="List all stargazers",
            version="v1.0",
            toolkit_versions={"github": "v2.0"},
            modifiers={"cache": True}
        )
        
        assert result["successful"] is True
        assert "data" in result
        assert "error" in result
    
    def test_execute_failure_response(self):
        """Test handling of failed execution response."""
        self.mock_composio.tools._execution_result = {
            "successful": False,
            "data": None,
            "error": "API rate limit exceeded"
        }
        
        result = self.adapter.execute(
            slug="GITHUB_LIST_STARGAZERS",
            arguments={"owner": "test", "repo": "test"}
        )
        
        assert result["successful"] is False
        assert result["data"] is None
        assert result["error"] == "API rate limit exceeded"
    
    def test_execute_response_normalization(self):
        """Test normalization of execution response."""
        # Test with response that has the expected format
        mock_response = {
            "successful": True,
            "data": {"result": "test"},
            "error": None
        }
        
        self.mock_composio.tools._execution_result = mock_response
        
        result = self.adapter.execute(
            slug="TEST_TOOL",
            arguments={"test": "value"}
        )
        
        assert result["successful"] is True
        assert result["data"] == {"result": "test"}
        assert result["error"] is None
    
    def test_execute_response_dict_conversion(self):
        """Test response conversion from TypedDict-like objects."""
        # Mock a response that has a copy method and can be converted to dict
        class MockTypedDictResponse:
            def __init__(self):
                self._data = {
                    "successful": True,
                    "data": {"converted": "response"},
                    "error": None
                }
            
            def copy(self):
                return self._data.copy()
            
            def get(self, key, default=None):
                return self._data.get(key, default)
            
            def __iter__(self):
                return iter(self._data)
            
            def __getitem__(self, key):
                return self._data[key]
        
        mock_response = MockTypedDictResponse()
        self.mock_composio.tools._execution_result = mock_response
        
        result = self.adapter.execute(
            slug="TEST_TOOL",
            arguments={"test": "value"}
        )
        
        assert result["successful"] is True
        assert result["data"] == {"converted": "response"}
        assert result["error"] is None
    
    def test_execute_missing_response_keys(self):
        """Test handling of response with missing keys."""
        self.mock_composio.tools._execution_result = {"data": "some data"}  # Missing successful and error
        
        result = self.adapter.execute(
            slug="TEST_TOOL",
            arguments={"test": "value"}
        )
        
        assert result["successful"] is False  # Default for missing successful
        assert result["data"] == "some data"
        assert result["error"] is None  # Default for missing error
    
    def test_execute_response_coercion_failure(self):
        """Test handling when response coercion to dict fails."""
        # Create a mock response that looks like it needs conversion but works
        class MockResponse:
            def get(self, key, default=None):
                return {
                    "successful": True,
                    "data": {"original": "response"},
                    "error": None
                }.get(key, default)
            
            def copy(self):
                raise Exception("Conversion failed")
            
            # Make hasattr return True for copy
            def __getattr__(self, name):
                if name == "copy":
                    return self.copy
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        mock_response = MockResponse()
        self.mock_composio.tools._execution_result = mock_response
        
        with patch('agentflow.adapters.tools.composio_adapter.logger') as mock_logger:
            result = self.adapter.execute(
                slug="TEST_TOOL",
                arguments={"test": "value"}
            )
            
            # Should still work with the original response
            assert result["successful"] is True
            assert result["data"] == {"original": "response"}
            assert result["error"] is None
            mock_logger.debug.assert_called_once()


class TestComposioAdapterIntegration:
    """Integration tests for ComposioAdapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_composio = MockComposioSDK()
        
        # Patch the availability and Composio class
        self.composio_patcher = patch('agentflow.adapters.tools.composio_adapter.HAS_COMPOSIO', True)
        self.class_patcher = patch('agentflow.adapters.tools.composio_adapter.Composio', return_value=self.mock_composio)
        
        self.composio_patcher.start()
        self.class_patcher.start()
        
        self.adapter = ComposioAdapter(
            api_key="test-key", provider="test-provider", 
            file_download_dir="/test/dir", 
            toolkit_versions=None
        )
    
    def teardown_method(self):
        """Clean up patches."""
        self.composio_patcher.stop()
        self.class_patcher.stop()
    
    def test_full_workflow_list_and_execute(self):
        """Test complete workflow of listing tools and executing one."""
        # List tools
        tools = self.adapter.list_tools_for_llm(user_id="test-user")
        assert len(tools) > 0
        
        # Get tool name
        tool_name = tools[0]["function"]["name"]
        assert tool_name == "GITHUB_LIST_STARGAZERS"
        
        # Execute the tool
        result = self.adapter.execute(
            slug=tool_name,
            arguments={"owner": "test", "repo": "test"},
            user_id="test-user"
        )
        
        assert result["successful"] is True
        assert result["data"] is not None
    
    def test_adapter_with_different_configurations(self):
        """Test adapter with different initialization configurations."""
        configs = [
            {},  # Default config
            {"api_key": "custom-key"},
            {"provider": "custom-provider"},
            {"file_download_dir": "/custom/dir"},
            {"toolkit_versions": {"custom": "v1.0"}},
            {  # All custom
                "api_key": "full-key",
                "provider": "full-provider", 
                "file_download_dir": "/full/dir",
                "toolkit_versions": {"full": "v2.0"}
            }
        ]
        
        for config in configs:
            with patch('agentflow.adapters.tools.composio_adapter.Composio') as mock_composio_class:
                mock_composio_class.return_value = self.mock_composio
                
                adapter = ComposioAdapter(**config)
                
                # Should be able to list tools
                tools = adapter.list_tools_for_llm(user_id="test")
                assert isinstance(tools, list)
                
                # Should be able to execute
                result = adapter.execute(slug="TEST", arguments={})
                assert "successful" in result
    
    def test_error_handling_throughout_workflow(self):
        """Test error handling in various parts of the workflow."""
        # Test tool listing with SDK errors - it should reraise the exception
        self.mock_composio.tools.get = Mock(side_effect=Exception("SDK Error"))
        self.mock_composio.tools.get_raw_composio_tools = Mock(return_value=[])
        
        with pytest.raises(Exception, match="SDK Error"):
            self.adapter.list_tools_for_llm(user_id="test")
        
        # Test execution with SDK errors
        self.mock_composio.tools.execute = Mock(side_effect=Exception("Execution Error"))
        
        with pytest.raises(Exception, match="Execution Error"):
            self.adapter.execute(slug="TEST", arguments={})
    
    def test_logging_behavior(self):
        """Test that appropriate logging occurs."""
        with patch('agentflow.adapters.tools.composio_adapter.logger') as mock_logger:
            # Test with raw tools that will cause logging
            class FaultyTool:
                @property
                def slug(self):
                    raise Exception("Test exception")
            
            faulty_tool = FaultyTool()
            # Add both faulty and good tools
            self.mock_composio.tools._raw_tools = [faulty_tool, MockRawTool()]
            # Empty mock tools to trigger fallback to raw tools
            self.mock_composio.tools._mock_tools = []
            
            self.adapter.list_tools_for_llm(user_id="test")
            
            # Should have logged warning message about failing to map tool schema
            assert mock_logger.warning.called
    
    def test_adapter_state_consistency(self):
        """Test that adapter maintains consistent state across operations."""
        # Verify adapter state doesn't change between operations
        initial_composio = self.adapter._composio
        
        self.adapter.list_tools_for_llm(user_id="test1")
        assert self.adapter._composio is initial_composio
        
        self.adapter.list_raw_tools_for_llm(tool_slugs=["TEST"])
        assert self.adapter._composio is initial_composio
        
        self.adapter.execute(slug="TEST", arguments={})
        assert self.adapter._composio is initial_composio


class TestComposioAdapterEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_composio = MockComposioSDK()
        
        # Patch the availability and Composio class
        self.composio_patcher = patch('agentflow.adapters.tools.composio_adapter.HAS_COMPOSIO', True)
        self.class_patcher = patch('agentflow.adapters.tools.composio_adapter.Composio', return_value=self.mock_composio)
        
        self.composio_patcher.start()
        self.class_patcher.start()
        
        self.adapter = ComposioAdapter(
            api_key="",
            provider=None,
            file_download_dir="",
            toolkit_versions=None
        )
    
    def teardown_method(self):
        """Clean up patches."""
        self.composio_patcher.stop()
        self.class_patcher.stop()
    
    def test_large_tool_lists(self):
        """Test handling of large tool lists."""
        # Create a large list of tools
        large_tool_list = []
        for i in range(100):
            large_tool_list.append({
                "type": "function",
                "function": {
                    "name": f"TOOL_{i}",
                    "description": f"Tool number {i}",
                    "parameters": {"type": "object", "properties": {}}
                }
            })
        
        self.mock_composio.tools._mock_tools = large_tool_list
        
        tools = self.adapter.list_tools_for_llm(user_id="test")
        assert len(tools) == 100
    
    def test_empty_arguments_execution(self):
        """Test execution with empty arguments."""
        result = self.adapter.execute(slug="TEST_TOOL", arguments={})
        
        assert "successful" in result
        assert "data" in result
        assert "error" in result
    
    def test_none_values_in_parameters(self):
        """Test handling of None values in various parameters."""
        # Test tool listing with None values
        tools = self.adapter.list_tools_for_llm(
            user_id="test",
            tool_slugs=None,
            toolkits=None,
            search=None,
            scopes=None,
            limit=None
        )
        assert isinstance(tools, list)
        
        # Test raw tool listing with None values
        raw_tools = self.adapter.list_raw_tools_for_llm(
            tool_slugs=None,
            toolkits=None,
            search=None,
            scopes=None,
            limit=None
        )
        assert isinstance(raw_tools, list)
        
        # Test execution with None values
        result = self.adapter.execute(
            slug="TEST",
            arguments={},
            user_id=None,
            connected_account_id=None,
            custom_auth_params=None,
            custom_connection_data=None,
            text=None,
            version=None,
            toolkit_versions=None,
            modifiers=None
        )
        assert isinstance(result, dict)
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        # Test with unicode tool names and descriptions
        unicode_tool = {
            "type": "function",
            "function": {
                "name": "测试工具",
                "description": "Un outil de test avec des caractères spéciaux: àáâãäåæçèéêë",
                "parameters": {"type": "object", "properties": {}}
            }
        }
        
        self.mock_composio.tools._mock_tools = [unicode_tool]
        
        tools = self.adapter.list_tools_for_llm(user_id="test")
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "测试工具"
        assert "caractères spéciaux" in tools[0]["function"]["description"]
        
        # Test execution with unicode arguments
        result = self.adapter.execute(
            slug="测试工具",
            arguments={"param": "värde med åäö"}
        )
        assert isinstance(result, dict)