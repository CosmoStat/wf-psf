"""Generic registry utility.

Provides a reusable implementation of the Registry pattern for
registering and retrieving objects by a unique key.

The generic registry serves as a foundation for building
domain-specific registries throughout WaveDiff (e.g. quality metrics,
rejection policies, etc.), while keeping registration and lookup behaviour
consistent across the codebase.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Registry(Generic[K, V]):
    """Generic registry implementation.

    Stores objects under unique keys and provides registration,
    lookup, and iteration operations.

    The registry is intended to be specialised by domain-specific
    registries rather than used directly by application code.
    """

    def __init__(self):
        """Initialize the registry with an empty internal store."""
        self._store: dict[K, V] = {}

    def register(self, key: K, value: V) -> None:
        """Register a value under a unique key.

        Parameters
        ----------
        key : Key
            Unique identifier used to register the value.
        value : Value
            Object to register.

        Raises
        ------
        ValueError
            If the key is already registered.
        """
        if key in self._store:
            raise ValueError(f"Key '{key}' is already registered.")
        self._store[key] = value

    def get(self, key: K) -> V:
        """Retrieve the value registered under a given key.

        Parameters
        ----------
        key : Key
            Unique identifier for the registered object.

        Returns
        -------
        V
            Registered object.

        Raises
        ------
        KeyError
            If the key is not present in the registry.

        """
        try:
            return self._store[key]
        except KeyError:
            raise KeyError(f"Key '{key}' not found in registry.") from None

    def unregister(self, key: K) -> None:
        """Remove a key:value pair from the registry.

        Parameters
        ----------
        key : K
            Unique identifier of the item to remove.

        Raises
        ------
        KeyError
          If key is not present in registry.

        """
        if key not in self._store:
            raise KeyError(f"Key '{key}' not found in registry.")
        del self._store[key]

    def __contains__(self, key: K) -> bool:
        """Return whether a key is registered.

        Parameter
        ---------
        key : K
            Key to test.

        Returns
        -------
        bool
            True if the key is present in the registry, otherwise False.
        """
        return key in self._store

    def __iter__(self):
        """Iterate over registered (key, value) pairs."""
        return iter(self._store.items())

    def keys(self):
        """Return a view of the registered keys."""
        return self._store.keys()

    def values(self):
        """Return a view of the registered values."""
        return self._store.values()

    def items(self):
        """Return a view of the registered (key, value) pairs."""
        return self._store.items()
