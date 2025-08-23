"""Dependency injection system for PyAgenity graph execution.

This module provides a container for managing dependencies that can be injected
into node functions during graph execution.
"""

import logging
from typing import Any, TypeVar


T = TypeVar("T")

logger = logging.getLogger(__name__)


class DependencyContainer:
    """Container for managing dependencies that can be injected into graph nodes.

    This allows users to register dependencies once when creating the graph
    and have them automatically available in all node functions.

    Example:
        container = DependencyContainer()
        container.register("database", my_database_connection)
        container.register("logger", my_logger)

        # Later in node functions:
        def my_node(state, config, database: InjectDep[Database] = None):
            # database will be automatically injected
            database.query("SELECT * FROM users")
    """

    def __init__(self):
        self._dependencies: dict[str, Any] = {}

    def register(self, name: str, dependency: Any) -> None:
        """Register a dependency with a given name.

        Args:
            name: The name to register the dependency under
            dependency: The dependency instance to register
        """
        logger.debug(
            "Registering dependency '%s' of type %s",
            name,
            type(dependency).__name__,
        )
        self._dependencies[name] = dependency

    def get(self, name: str) -> Any:
        """Get a dependency by name.

        Args:
            name: The name of the dependency to retrieve

        Returns:
            The dependency instance

        Raises:
            KeyError: If the dependency is not registered
        """
        if name not in self._dependencies:
            error_msg = f"Dependency '{name}' not registered"
            logger.error(error_msg)
            raise KeyError(error_msg)
        logger.debug("Retrieved dependency '%s'", name)
        return self._dependencies[name]

    def has(self, name: str) -> bool:
        """Check if a dependency is registered.

        Args:
            name: The name of the dependency to check

        Returns:
            True if the dependency is registered, False otherwise
        """
        logger.debug("Checking if dependency '%s' is registered", name)
        return name in self._dependencies

    def unregister(self, name: str) -> None:
        """Unregister a dependency.

        Args:
            name: The name of the dependency to unregister
        """
        if name in self._dependencies:
            del self._dependencies[name]

        logger.debug("Unregistered dependency '%s'", name)

    def list_dependencies(self) -> list[str]:
        """List all registered dependency names.

        Returns:
            List of registered dependency names
        """
        res = list(self._dependencies.keys())
        logger.debug("Listing all registered dependencies: %s", res)
        return res

    def clear(self) -> None:
        """Clear all registered dependencies."""
        self._dependencies.clear()
        logger.debug("Cleared all registered dependencies")

    def copy(self) -> "DependencyContainer":
        """Create a copy of this container.

        Returns:
            A new DependencyContainer with the same dependencies
        """
        new_container = DependencyContainer()
        new_container._dependencies = self._dependencies.copy()
        logger.debug("Created a copy of the dependency container")
        return new_container
