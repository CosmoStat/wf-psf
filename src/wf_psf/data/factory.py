"""
Factory module for creating and normalizing data adapters.

This module defines the ``DataAdapterFactory``, which constructs ``DataAdapter``
instances from a variety of dataset formats, including dictionaries,
dataclasses, ``LoadedDataset`` instances, or objects with attributes exposing
numpy arrays. It also integrates dataset normalization through the ``DataEnvelope``
and utility routines in ``data_utils``.

The module defines a protocol (``SupportsParams``) to allow
external APIs to pass parameter containers in a generic way,
supporting dataclasses, custom objects, or dictionaries.

Key features:

- Automatic detection of dataset structure (train/test/complete) and conversion
  to ``LoadedDataset`` for downstream processing.
- Normalization and validation of dataset parameters via ``normalize_data_envelope``.
- Optional metadata extraction when available in input objects.
- Integration with ``TensorFlowDatasetConverter`` for TF-ready dataset pipelines.
- Lightweight dataset introspection utilities for in-memory datasets and canonical keys.
- Logging to provide insight into dataset resolution and loading steps.

Author: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from dataclasses import dataclass, is_dataclass, fields
from typing import Any, Optional, Union
from wf_psf.data.data_adapter import DataAdapter, LoadedDataset
from wf_psf.data.npy_dataset_loader import NpyDatasetLoader
from wf_psf.data.tensorflow_converter import TensorFlowDatasetConverter
from typing import Protocol, runtime_checkable
from wf_psf.utils.read_config import RecursiveNamespace
import logging

logger = logging.getLogger(__name__)


# Define protocols to allow external APIs to be used
@runtime_checkable
class SupportsParams(Protocol):
    """Protocol for dataset objects containing parameters.

    This protocol represents objects that expose a ``params`` attribute
    containing dataset parameters. This allows dataclasses, custom objects, and other parameter containers
    to be accepted by the data adapter API.

    Attributes
    ----------
    params : Any
        An object (e.g., dict, structured namespace, etc) containing dataset-specific parameters.
    """

    params: Any


# Define a union type for all acceptable data formats that include parameters
DataInput = Union[dict[str, Any], SupportsParams]
ParamsType = Union[dict[str, Any], RecursiveNamespace]


@dataclass
class DataEnvelope:
    """
    Encapsulates separated dataset, parameters and metadata.

    Attributes
    ----------
    data : Optional[Any]
        The actual dataset (e.g., ``LoadedDataset``, ``dict``, ``dataclass``). Can be ``None`` if input is just params.
    params : ParamsType
        Configuration parameters used to resolve and load the dataset. Required for adapter construction.
    metadata : Optional[dict] = None
        Ancillary information about the dataset (IDs, units, provenance, etc.).
        Defaults to None if not present in input.
    """

    data: Optional[Any]
    params: Any
    metadata: Optional[dict] = None


def normalize_data_envelope(
    obj: Any, field_name: str = "params", metadata_name: str = "metadata"
) -> DataEnvelope:
    """Normalize data envelope.

    Normalize an input object into a ``DataEnvelope`` by extracting named
    parametric fields and metadata. Supports dataclasses, dictionaries,
    and generic objects with attributes.

    Parameters
    ----------
    obj : Any
        Input object containing dataset, parameters, and optionally metadata.
    field_name : str, default "params"
        Name of the field to extract as parameters.
    metadata_name : str, default "metadata"
        Name of the field to extract as metadata, if present.

    Returns
    -------
    DataEnvelope
        Object containing separated data, parameters, and metadata.

    Notes
    -----
    - The ``params`` field is optional, but may be required by downstream components (e.g. the factory) to resolve how the dataset should be constructed (in-memory vs. file-based loading).
    - The ``metadata`` field is optional and ignored if not present.

    """
    # -----------------------
    # Dataclass input
    # -----------------------
    if is_dataclass(obj):
        params = getattr(obj, field_name, None)
        metadata = getattr(obj, metadata_name, None)
        data_fields = [
            f.name for f in fields(obj) if f.name not in (field_name, metadata_name)
        ]
        data = None if not data_fields else {f: getattr(obj, f) for f in data_fields}

    # -----------------------
    # Dictionary input
    # -----------------------
    elif isinstance(obj, dict):
        obj_copy = dict(obj)

        # Extract params
        params = obj_copy.pop(field_name, None)

        # Extract metadata
        metadata = obj_copy.pop(metadata_name, None)

        # Rest is data or None
        data = obj_copy or None

    # -----------------------
    # Generic object with attributes
    # -----------------------
    elif hasattr(obj, field_name):
        params = getattr(obj, field_name, None)
        metadata = getattr(obj, metadata_name, None)
        data_attrs = {
            k: v
            for k, v in obj.__dict__.items()
            if k not in (field_name, metadata_name)
        }
        data = None if not data_attrs else data_attrs

    else:
        raise TypeError(f"Unsupported input type for data normalization: {type(obj)}")

    return DataEnvelope(data=data, params=params, metadata=metadata)


class DataAdapterFactory:
    """Factory for creating DataAdapters from various dataset formats."""

    @staticmethod
    def build(data):
        """
        Create a DataAdapter.

        Parameters
        ----------
        data : object
            The dataset to be adapted. Can be:

            - A ``LoadedDataset`` instance
            - A ``dataclass`` with numpy arrays (e.g., train/test containers, parameters or shallow complete)
            - A ``dict`` containing 'train', 'test', or 'complete' keys with numpy arrays
            - An ``object`` with attributes that are numpy arrays (like your train/test containers)
            The factory will automatically detect the structure and convert it into a ``LoadedDataset``.

        Returns
        -------
        DataAdapter
        """
        dataset, params, metadata = DataAdapterFactory._resolve_dataset(data=data)
        converter = TensorFlowDatasetConverter()
        return DataAdapter(
            dataset=dataset, params=params, metadata=metadata, converter=converter
        )

    @staticmethod
    def _resolve_dataset(
        data: DataInput,
    ) -> tuple[LoadedDataset, ParamsType, Optional[Any]]:
        """Resolve dataset.

        Resolution proceeds in two stages:

        1. Determine whether the dataset should be loaded from disk or treated as
        in-memory, based solely on the structure of `params`.

        2. Validate consistency between `params` and the provided dataset, then
        construct a `LoadedDataset` accordingly.

        Parameters
        ----------
        data : DataInput
            Input dataset in any supported format (dict, dataclass, or object with
            attributes), optionally containing associated parameters.


        Returns
        -------
        tuple
            A tuple containing the loaded dataset, data parameters, and metadata (optional).

        Notes
        -----
        - Dataset resolution is driven by ``params``.
        - If file-based configuration (e.g. ``file`` or ``data_dir``) is detected, the dataset is loaded from disk.
        - Otherwise, the dataset is assumed to be provided in memory.

        """
        # Normalise data
        envelope = normalize_data_envelope(data)
        dataset, params, metadata = (
            envelope.data,
            envelope.params,
            envelope.metadata,
        )

        # Check if data and params are None
        if dataset is None and params is None:
            raise ValueError("No data or configuration parameters provided.")

        # Determine dataset source (in-memory vs. file-based) from params.
        # Data presence is validated against that inferred intent.
        in_memory = _is_in_memory(params)

        # Case A — In-memory data
        if in_memory:
            if dataset is None:
                raise ValueError(
                    "Parameters indicate in-memory data (no 'file'/'data_dir' found), "
                    "but no dataset was provided."
                )
            return (_build_loaded_dataset(dataset), params, metadata)

        # Case B — Load from disk
        if params is None:
            raise ValueError("Missing dataset parameters; cannot load data from disk.")

        logger.info(
            "No in-memory data detected. Attempting to load dataset from files "
            "based on provided parameters."
        )
        return (DataAdapterFactory._load_dataset(params), params, metadata)

    @staticmethod
    def _load_dataset(params) -> LoadedDataset:
        """Load dataset.

        Load dataset using configuration parameters.

        Parameters
        ----------
        params : RecursiveNamespace
            A recursive namespace object containing dataset configuration parameters needed to load data from disc.

        Returns
        -------
        LoadedDataset
            Dataset container populated from the provided configuration.

        """
        data_cfg = params

        # -------------------------
        # Case 1: Split configuration
        # -------------------------
        if hasattr(data_cfg, "train") and hasattr(data_cfg, "test"):
            train_loader = NpyDatasetLoader(data_cfg.train)
            test_loader = NpyDatasetLoader(data_cfg.test)

            train_loader.load()
            test_loader.load()

            return LoadedDataset(
                train=train_loader.dataset,
                test=test_loader.dataset,
            )

        # -------------------------
        # Case 2: Complete configuration
        # -------------------------
        elif hasattr(data_cfg, "complete"):
            complete_loader = NpyDatasetLoader(data_cfg.complete)
            complete_loader.load()

            return LoadedDataset(
                complete=complete_loader.dataset,
            )
        # -------------------------
        # Case 3: Shallow configuration
        # -------------------------
        elif hasattr(data_cfg, "file"):
            shallow_loader = NpyDatasetLoader(data_cfg)
            shallow_loader.load()

            return LoadedDataset(
                complete=shallow_loader.dataset,
            )
        else:
            raise ValueError(
                "Cannot determine dataset source from configuration. Please provide either 'train' and 'test' configs or a 'file' config."
            )


def _is_in_memory(params: Optional[ParamsType]) -> bool:
    """Determine whether the dataset is already held in memory.

    Inspects ``params`` for the presence of ``file`` and ``data_dir`` keys across
    all three supported config shapes: shallow (keys at the top level), complete
    (keys nested under a ``complete`` block), and split (keys nested under
    ``train`` and ``test`` blocks).

    Parameters
    ----------
    params : Optional[ParamsType]
        Dataset configuration parameters, either as a plain ``dict`` or a
        dataclass. May be ``None`` if the caller supplied raw in-memory data
        with no associated configuration. The function inspects both the
        top-level structure and any nested ``complete`` block for ``file``
        and ``data_dir`` keys.

    Returns
    -------
    bool
        ``True`` if the dataset is in memory and no file loading is required,
        ``False`` if ``params`` contains a ``file`` or ``data_dir`` pointer
        indicating that data must be loaded from disk.

    """
    if params is None:
        logger.warning("Params field is None, assuming data is in memory.")
        return True

    def has_file_pointer(obj) -> bool:
        """Return True if obj contains 'file' or 'data_dir'."""
        d = (
            obj
            if isinstance(obj, dict)
            else vars(obj)
            if hasattr(obj, "__dict__")
            else {}
        )
        return "file" in d or "data_dir" in d

    top = (
        params
        if isinstance(params, dict)
        else vars(params)
        if hasattr(params, "__dict__")
        else {}
    )

    # Shallow config: file/data_dir at the top level
    if has_file_pointer(top):
        logger.info("Detected file and data_dir fields, assuming data is not in memory.")
        return False

    # Complete (non-split) config: file/data_dir nested under 'complete'
    if "complete" in top and has_file_pointer(top["complete"]):
        logger.info("Detected file and data_dir fields for complete dataset configuration, assuming data is not in memory.")
        return False

    # Split config: file/data_dir nested under 'train' and 'test'
    if "train" in top and has_file_pointer(top["train"]):
        logger.info("Detected file and data_dir fields for split dataset configuration, assuming data is not in memory.")
        return False
    if "test" in top and has_file_pointer(top["test"]):
        logger.info("Detected file and data_dir fields for split dataset configuration, assuming data is not in memory.")
        return False

    return True


def _build_loaded_dataset(dataset: Any) -> LoadedDataset:
    """Construct a LoadedDataset from an in-memory dataset.

    Inspects the structure of ``dataset`` to determine whether it represents
    a split (train/test) or complete (non-split) dataset, and wraps it in the
    appropriate ``LoadedDataset`` form. If the structure cannot be determined,
    the entire object is treated as the complete dataset and a warning is logged.

    Parameters
    ----------
    dataset : Any
        An in-memory dataset containing numpy arrays, provided as a plain
        ``dict``, a dataclass with named fields, or an opaque object. Expected
        to hold arrays under recognised keys (``'train'``, ``'test'``, or
        ``'complete'``), though unrecognised structures are handled with a
        fallback.

    Returns
    -------
    LoadedDataset
        A structured dataset container populated according to the detected shape of
        ``dataset``:

        - ``LoadedDataset(train=..., test=...)`` if both ``'train'`` and
          ``'test'`` keys are present.
        - ``LoadedDataset(complete=...)`` if a ``'complete'`` key is present.
        - ``LoadedDataset(complete=dataset)`` as a fallback for flat or
          unrecognised structures, with a logged warning.
    """
    # Normalise to a plain dict so the rest of the logic is uniform,
    # regardless of whether the caller passed a dict or a dataclass.
    if isinstance(dataset, dict):
        d = dataset
    elif hasattr(dataset, "__dict__"):
        d = vars(dataset)
    else:
        # Opaque object — treat the whole thing as the complete array.
        logger.warning(
            "Cannot inspect dataset structure; treating entire object as 'complete'."
        )
        return LoadedDataset(complete=dataset)

    if "train" in d and "test" in d:
        logger.info("In-memory split dataset detected (train/test).")
        return LoadedDataset(train=d["train"], test=d["test"])

    if "complete" in d:
        logger.info("In-memory complete dataset detected.")
        return LoadedDataset(complete=d["complete"])

    # Fallback: shallow / flat structure with no recognised keys.
    logger.warning(
        "In-memory dataset has no 'complete' or 'train'/'test' keys; "
        "treating entire dataset as 'complete'."
    )
    return LoadedDataset(complete=dataset)
