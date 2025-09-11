"""Tests for the config module."""

import pytest
from pydantic import ValidationError
from pyagenity.graph.config import Config


class TestConfig:
    """Test the Config class."""

    def test_config_creation(self):
        """Test creating a Config instance."""
        config = Config(api_key="test_key", api_url="https://api.example.com", timeout=30)
        assert config.api_key == "test_key"
        assert config.api_url == "https://api.example.com"
        assert config.timeout == 30

    def test_config_from_dict(self):
        """Test creating a Config from a dictionary."""
        data = {"api_key": "dict_key", "api_url": "https://dict.example.com", "timeout": 60}
        config = Config(**data)
        assert config.api_key == "dict_key"
        assert config.api_url == "https://dict.example.com"
        assert config.timeout == 60

    def test_config_validation(self):
        """Test Config validation."""
        # Valid config
        config = Config(api_key="valid", api_url="https://valid.com", timeout=10)
        assert config is not None

        # Invalid - missing required field
        with pytest.raises(ValueError):
            Config(api_url="https://test.com", timeout=30)  # missing api_key

        with pytest.raises(ValueError):
            Config(api_key="test", timeout=30)  # missing api_url

        with pytest.raises(ValueError):
            Config(api_key="test", api_url="https://test.com")  # missing timeout

    def test_config_model_dump(self):
        """Test Config model_dump method."""
        config = Config(api_key="dump_key", api_url="https://dump.example.com", timeout=45)
        data = config.model_dump()
        assert data["api_key"] == "dump_key"
        assert data["api_url"] == "https://dump.example.com"
        assert data["timeout"] == 45

    def test_config_json_serialization(self):
        """Test Config JSON serialization."""
        config = Config(api_key="json_key", api_url="https://json.example.com", timeout=90)
        json_str = config.model_dump_json()
        assert "json_key" in json_str
        assert "https://json.example.com" in json_str
        assert "90" in json_str

    def test_config_field_types(self):
        """Test Config field types."""
        config = Config(api_key="type_test", api_url="https://type.example.com", timeout=120)

        # Check field types
        assert isinstance(config.api_key, str)
        assert isinstance(config.api_url, str)
        assert isinstance(config.timeout, int)

    def test_config_immutability(self):
        """Test that Config fields are properly validated during creation."""
        # Test that invalid types are rejected during creation
        with pytest.raises(ValidationError):
            Config(api_key=123, api_url="https://test.com", timeout=30)  # api_key should be string

        with pytest.raises(ValidationError):
            Config(
                api_key="test", api_url="https://test.com", timeout="not_int"
            )  # timeout should be int

    def test_config_equality(self):
        """Test Config equality comparison."""
        config1 = Config(api_key="equal", api_url="https://equal.com", timeout=20)
        config2 = Config(api_key="equal", api_url="https://equal.com", timeout=20)
        config3 = Config(api_key="different", api_url="https://equal.com", timeout=20)

        assert config1 == config2
        assert config1 != config3

    def test_config_repr(self):
        """Test Config string representation."""
        config = Config(api_key="repr_test", api_url="https://repr.example.com", timeout=25)
        repr_str = repr(config)
        assert "Config" in repr_str
        assert "repr_test" in repr_str
        assert "https://repr.example.com" in repr_str
        assert "25" in repr_str
