"""
Data utilities and lightweight runtime data structures.

Provides lightweight dataset containers, runtime conversion contexts,
and helper utilities used throughout the dataset normalization and
preprocessing pipeline.

This module includes:

- Dictionary-like dataset container abstractions
- Dataset inspection and normalization helpers
- Runtime conversion context objects used during field-level processing
- Domain-specific preprocessing contexts (e.g. SED processing)

These utilities support schema-driven dataset conversion workflows used
by training, validation, and inference pipelines.

Notes
-----
The conversion context system is intentionally extensible to support
additional scientific domains and instrument pipelines beyond the
current Euclid-specific workflows. Future extensions
may include dedicated contexts for:

- PSF modeling
- Instrument calibration
- Detector noise simulation

Authors
-------
Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from __future__ import annotations
from collections.abc import MutableMapping
from dataclasses import dataclass, is_dataclass, fields
from typing import Any, Optional 

from wf_psf.sims.psf_simulator import PSFSimulator


@dataclass(frozen=True)
class SEDContext:
    """
    Context object containing parameters required for SED processing.

    This context encapsulates all runtime dependencies needed for
    spectral energy distribution (SED) transformations within the
    dataset conversion pipeline.

    Parameters
    ----------
    simPSF : Any
        PSF simulator instance used during SED processing. This object
        is responsible for modelling instrument response effects applied
        to spectral data.
    n_bins_lambda : int
        Number of wavelength bins used for discretizing the SED during
        conversion.
    """

    simPSF: PSFSimulator
    n_bins_lambda: int


@dataclass(frozen=True)
class ConversionContext:
    """
    Global runtime context for dataset conversion operations.

    This object aggregates optional domain-specific contexts required
    during dataset preprocessing and conversion. It is passed through
    the conversion pipeline and accessed by field-specific handlers.

    Currently, it contains an optional SED context used for spectral
    energy distribution processing.

    Design Note
    -----------
    This structure is intentionally extensible to support additional
    scientific domains beyond SED processing. Future extensions may
    include, for example:

    - PSFContext: instrument point spread function modeling
    - InstrumentContext: instrument-specific calibration and metadata
    - NoiseContext: detector noise or simulation noise models

    This design allows the framework to generalize beyond Euclid and
    support additional instruments or simulation pipelines without
    modifying the core converter logic.

    Parameters
    ----------
    seds : SEDContext or None
        Optional context required for SED-related field processing.
        If None, SED-dependent handlers should not be invoked.
    """

    seds: SEDContext | None = None


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
