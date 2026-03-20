"""
Factory module for creating and normalizing data adapters.

This module defines the DataAdapterFactory, which constructs DataAdapter
instances from a variety of dataset formats, including dictionaries,
dataclasses, LoadedDataset instances, or objects with attributes exposing
numpy arrays. It also integrates dataset normalization and metadata extraction
through the DataEnvelope and utility routines in `data_utils`.

The module defines protocols (`SupportsParams`, `SupportsMetadata`) to allow
external APIs to pass parameter and metadata containers in a generic way,
supporting dataclasses, custom objects, or dictionaries.

Key features:
- Automatic detection of dataset structure (train/test/complete) and conversion
  to LoadedDataset for downstream processing.
- Normalization and validation of dataset parameters via `normalize_data_envelope`.
- Integration with TensorFlowDatasetConverter for TF-ready dataset pipelines.
- Lightweight dataset introspection utilities for numpy arrays and canonical keys.
- Logging to provide insight into dataset resolution and loading steps.

Author: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from dataclasses import dataclass, is_dataclass, fields
from typing import Any, Optional, Union
from wf_psf.data.data_adapter import DataAdapter, LoadedDataset
from wf_psf.data.data_utils import DatasetUtils
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


@runtime_checkable
class SupportsMetadata(Protocol):
    """Protocol for dataset objects exposing metadata.

    Objects satisfying this protocol must expose a ``metadata`` attribute
    containing a structured namespace of parameters.

    This allows external APIs to pass parameter containers without
    requiring a specific implementation type.

    Attributes
    ----------
    metadata : Any
        An object (e.g., dict, structured namespace, etc) containing dataset-specific metadata.
    """

    metadata: Any


# Define a union type for all acceptable data formats that include parameters
DataWithParams = Union[dict[str, Any], SupportsParams]
ParamsType = Union[dict[str, Any], RecursiveNamespace]


@dataclass
class DataEnvelope:
    """
    Encapsulates separated dataset, parameters and metadata.

    Attributes
    ----------
    data : Optional[Any]
        The actual dataset (e.g., LoadedDataset, dict, dataclass). Can be None if input is just params.
    params : ParamsType
        Configuration parameters.
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

    Normalize an input object into a DataEnvelope by extracting named
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
    The `params` field is optional. If not provided, downstream components
    (e.g. DataAdapter) will rely on default configuration values.

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
            - A LoadedDataset instance
            - A dataclass with numpy arrays (e.g., train/test containers, parameters or shallow complete)
            - A dict containing 'train', 'test', or 'complete' keys with numpy arrays
            - An object with attributes that are numpy arrays (like your train/test containers)
            The factory will automatically detect the structure and convert it into a LoadedDataset.

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
        data: DataWithParams,
    ) -> tuple[LoadedDataset, ParamsType, Optional[Any]]:
        """Resolve dataset.

        Determine whether to use in-memory data or load from files.

        Parameters
        ----------
        data : DataWithParams
            Union type of data in various formats containing associated parameters or metadata.

        Returns
        -------
        tuple
            A tuple containing the loaded dataset, data parameters, and metadata.

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

        # Case A - Check if in-memory data provided with numpy arrays
        if DatasetUtils._contains_numpy(dataset):
            logger.info(
                "In-memory data with numpy arrays detected. Constructing LoadedDataset directly."
            )
            # Construct a LoadedDataset from in-memory dataset
            # Handle different structures: complete, split
            if "complete" in dataset:
                return (
                    LoadedDataset(complete=dataset["complete"]),
                    params,
                    metadata,
                )
            elif "train" in dataset and "test" in dataset:
                return (
                    LoadedDataset(train=dataset["train"], test=dataset["test"]),
                    params,
                    metadata,
                )
            else:
                # fallback for shallow data
                logger.warning(
                    "Data contains numpy arrays but does not have 'complete' or 'train/test' attributes. Attempting to treat entire data as 'complete'."
                )
                return (LoadedDataset(complete=dataset), params, metadata)

        # Case B — No in-memory data → use loader
        else:
            if params is None:
                # Raise error if params is None
                raise ValueError(
                    "Missing dataset parameters; cannot load data from disk."
                )

            # Proceed to load dataset from disk using provided parameters
            logger.info(
                "No in-memory data with numpy arrays detected. Attempting to load dataset from files based on provided parameters."
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
            Dataset contain

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
