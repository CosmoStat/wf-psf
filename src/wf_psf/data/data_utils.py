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
