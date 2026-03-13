"""
Data utilities.

Provides lightweight dataset containers and utilities for dataset
introspection, such as detecting NumPy arrays and converting heterogeneous
data structures into standardized containers.

These utilities support the dataset normalization and loading pipeline
used by the data adapter and factory components.

Authors
-------
Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from collections.abc import MutableMapping
from dataclasses import is_dataclass, fields
from typing import Any, Optional
import numpy as np


class DatasetContainer(MutableMapping):
    """
    Lightweight container for structured dataset data.

    Stores data internally as a dictionary, while providing
    dictionary-style and attribute-style access for convenience.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary containing dataset tensors and metadata.

    Attributes
    ----------
    _data : dict[str, Any]
        Internal storage for dataset contents.

    Examples
    --------
    >>> container = DatasetContainer({'x': np.array([1, 2, 3]), 'y': np.array([4, 5, 6])})
    >>> container['x']
    array([1, 2, 3])
    >>> container.x
    array([1, 2, 3])
    >>> container.to_dict()
    {'x': array([1, 2, 3]), 'y': array([4, 5, 6])}
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getitem__(self, key):
        """Return the value stored under `key`."""
        return self._data[key]

    def __setitem__(self, key, value):
        """Store `value` at `key`."""
        self._data[key] = value

    def __delitem__(self, key):
        """Delete `key`."""
        del self._data[key]

    def __iter__(self):
        """Iterate over data."""
        return iter(self._data)

    def __len__(self):
        """Return length of data."""
        return len(self._data)

    def __getattr__(self, name: str):
        """Get data attribute."""
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        """Return data as dict."""
        return self._data


class DatasetUtils:
    """Utility functions for dataset introspection and structure detection.

    Provides helpers for detecting NumPy arrays and extracting information from
    heterogeneous data structures (e.g., dictionaries, dataclasses, objects with
    ``__dict__``, lists, and tuples).

    These utilities support dataset normalization and loading
    logic used by the data adapter factory.

    Attributes
    ----------
    MAX_RECURSION_DEPTH : int
        Maximum recursion depth used when traversing nested data structures to
        prevent infinite loops.
    """

    MAX_RECURSION_DEPTH = 5

    @staticmethod
    def _contains_numpy(obj: Any) -> bool:
        """
        Check whether an object contains at least one NumPy array.

        The check is performed recursively and supports
        nested dictionaries, dataclasses, object with
        ``__dict__``, lists, and tuples.

        Parameters
        ----------
        obj : Any
            Object to inspect.

        Returns
        -------
        bool
            True if at least one NumPy array is found within the object
            structure, False otherwise.
        """
        return DatasetUtils._contains_numpy_helper(
            obj, current_depth=0, max_depth=DatasetUtils.MAX_RECURSION_DEPTH
        )

    @staticmethod
    def _contains_numpy_helper(obj: Any, current_depth: int, max_depth: int) -> bool:
        """Check recursively if object contains NumPy array."""
        if current_depth > max_depth:
            return False  # Prevent infinite recursion

        if isinstance(obj, np.ndarray):
            return True

        if is_dataclass(obj):
            return any(
                DatasetUtils._contains_numpy_helper(
                    getattr(obj, f.name),
                    current_depth=current_depth + 1,
                    max_depth=max_depth,
                )
                for f in fields(obj)
            )

        if isinstance(obj, dict):
            return any(
                DatasetUtils._contains_numpy_helper(
                    v, current_depth=current_depth + 1, max_depth=max_depth
                )
                for v in obj.values()
            )

        if hasattr(obj, "__dict__"):
            return any(
                DatasetUtils._contains_numpy_helper(
                    v, current_depth=current_depth + 1, max_depth=max_depth
                )
                for v in vars(obj).values()
            )

        if isinstance(obj, (list, tuple)):
            return any(
                DatasetUtils._contains_numpy_helper(
                    v, current_depth=current_depth + 1, max_depth=max_depth
                )
                for v in obj
            )

        return False

    @staticmethod
    def to_container(obj) -> Optional[DatasetContainer]:
        """Convert an object to a ``DatasetContainer``.

        Transforms various dataset representations into a standardized
        :class:`DatasetContainer` used by downstream processing.

        Supported input types include dictionaries, dataclasses,
        objects with attributes, and existing ``DatasetContainer`` instances.

        Parameters
        ----------
        obj : Any
            Object representing dataset data.

        Returns
        -------
        DatasetContainer or None
            Structured container wrapping the dataset data.

        Raises
        ------
        TypeError
            If the input type is not supported.
        """
        if obj is None:
            return None

        if isinstance(obj, DatasetContainer):
            return obj

        if isinstance(obj, dict):
            return DatasetContainer(obj)

        if is_dataclass(obj):
            return DatasetContainer({f.name: getattr(obj, f.name) for f in fields(obj)})

        if hasattr(obj, "__dict__"):
            return DatasetContainer(vars(obj))

        raise TypeError(f"Unsupported dataset type: {type(obj)}")
