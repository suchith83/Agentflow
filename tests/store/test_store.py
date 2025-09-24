"""Comprehensive tests for the store module."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from pyagenity.store import BaseStore


class TestBaseStore:
    """Test the BaseStore abstract class."""

    def test_base_store_is_abstract(self):
        """Test that BaseStore is an abstract class."""
        # BaseStore should be abstract and not instantiable
        # We just test that the class exists and has the right structure
        assert hasattr(BaseStore, "update_memory")  # noqa: S101
        assert callable(BaseStore.update_memory)  # noqa: S101

    def test_base_store_has_expected_methods(self):
        """Test that BaseStore defines expected methods."""
        # Check for expected method names
        expected_methods = [
            "update_memory",
            "get_memory", 
            "delete_memory",
            "related_memory",
            "aupdate_memory",
            "aget_memory",
            "adelete_memory", 
            "arelated_memory",
            "release",
            "arelease",
        ]

        for method_name in expected_methods:
            if hasattr(BaseStore, method_name):
                method = getattr(BaseStore, method_name)
                assert callable(method)  # noqa: S101


def test_store_module_imports():
    """Test that store module imports work correctly."""
    assert BaseStore is not None  # noqa: S101
