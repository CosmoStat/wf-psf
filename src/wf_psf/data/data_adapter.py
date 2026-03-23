"""
Data Adapter.

This module manages dataset lifecycle transitions for the WF-PSF pipeline.

Two orthogonal state machines are maintained:

Structure state
---------------

    COMPLETE
        │
        │ split_data()
        ▼
    SPLIT
        │
        │ join_data()
        ▼
    COMPLETE


Representation state
--------------------

    NUMPY
        │
        │ convert_to_tensorflow()
        ▼
    TENSORFLOW


Glossary
--------
COMPLETE
    Dataset stored as a single container.

SPLIT
    Dataset stored as train/test subsets.

NUMPY
    Data stored as NumPy arrays.

TENSORFLOW
    Data stored as TensorFlow tensors.


Design principles
-----------------
- Structure and representation are orthogonal.
- All transitions are explicit and idempotent where possible.
- No training or model logic lives in this module.
- Dataset field names are canonicalized for downstream models.

The `DataAdapter` class manages these transitions while providing a
consistent interface for accessing dataset contents.

Authors: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from enum import Enum, auto
import numpy as np
from typing import Any, List, Optional
from wf_psf.data.constants import (
    DEFAULT_CANONICAL_KEYS,
    DEFAULT_SEED,
    DEFAULT_TRAIN_FRACTION,
)
from wf_psf.data.data_utils import DatasetContainer, to_container
from wf_psf.data.tensorflow_converter import TensorFlowDatasetConverter
import logging

logger = logging.getLogger(__name__)


class StructureState(Enum):
    """Structural state of the dataset.

    - COMPLETE: the dataset is complete and not split into train/test.
    - SPLIT: the dataset is split into train and test sets.
    """

    COMPLETE = auto()
    SPLIT = auto()


class RepresentationState(Enum):
    """Representation state of the dataset.

    - NUMPY: the dataset is represented as a NumPy array.
    - TENSORFLOW: the dataset is represented as a TensorFlow tensor.
    """

    NUMPY = auto()
    TENSORFLOW = auto()


class LoadedDataset:
    """Structured container for loaded dataset.

    Attributes
    ----------
    complete : dict, optional
        The complete dataset (if in COMPLETE state).
    train : dict, optional
        The training dataset (if in SPLIT state).
    test : dict, optional
        The test dataset (if in SPLIT state).
    """

    def __init__(
        self,
        complete: Optional[dict] = None,
        train: Optional[dict] = None,
        test: Optional[dict] = None,
    ):
        """Initialize the LoadedDataset with either complete or split data."""
        self.complete = complete
        self.train = train
        self.test = test

    def is_complete(self) -> bool:
        """Check if the dataset is in COMPLETE state."""
        return self.complete is not None

    def is_split(self) -> bool:
        """Check if the dataset is in SPLIT state."""
        return self.train is not None and self.test is not None


class DataAdapter:
    """Adapter for managing dataset structure and backend representation.

    The adapter provides a consistent interface to datasets regardless of
    whether they are stored as a complete dataset or as train/test splits,
    and whether the underlying representation is NumPy or TensorFlow.

    It also canonicalizes dataset fields to the names expected by
    downstream models.

    Notes
    -----
    Instances should be created via `DataAdapterFactory.build()`.
    """

    def __init__(
        self,
        dataset: LoadedDataset,
        converter: TensorFlowDatasetConverter,
        params: Optional[Any] = None,
        metadata: Optional[dict] = None,
    ):
        """Initialize the DataAdapter with a loaded dataset and optional converter and parameters.

        Parameters
        ----------
        dataset : LoadedDataset
            The loaded dataset containing either complete or data split into train and test (validation) datasets.
        converter : TensorFlowDatasetConverter
            A TensorFlowDatasetConverter object for transforming from NumPy to TensorFlow representations.
        params : optional
            Additional parameters for dataset management.
        metadata : optional
            Additional ancillary information for dataset management.

        Raises
        ------
        ValueError
            If the loaded dataset is empty or malformed (neither complete nor split).
        """
        if dataset is None:
            raise ValueError("LoadedDataset cannot be None.")

        self._params = params
        self._metadata = metadata
        self._converter = converter
        self._canonical_keys = getattr(params, "canonical_keys", DEFAULT_CANONICAL_KEYS)
        self._train_fraction = getattr(params, "train_fraction", DEFAULT_TRAIN_FRACTION)
        self._seed = getattr(params, "seed", DEFAULT_SEED)
        # Dataset containers for both representations
        # default empty
        self._train_data = None
        self._test_data = None
        self._complete_data = None

        # Determine Structure state
        if dataset.is_complete():
            self._structure_state = StructureState.COMPLETE
        elif dataset.is_split():
            self._structure_state = StructureState.SPLIT
        else:
            raise ValueError("Loaded dataset is empty or malformed")

        # Convert to containers
        self._initialize_structure(dataset)

        # Map user-provided keys to canonical keys
        self._canonicalize_initial_data()

        # Representation state always starts as NUMPY
        self._representation_state = RepresentationState.NUMPY

        # Placeholders for TensorFlow datasets
        self._train_tf = None
        self._test_tf = None
        self._complete_tf = None

    def _initialize_structure(self, dataset):
        """Convert to container."""
        if self._structure_state == StructureState.COMPLETE:
            self._complete_data = to_container(dataset.complete.copy())
            return

        self._train_data = to_container(dataset.train.copy())
        self._test_data = to_container(dataset.test.copy())

    def _resolve_target_field(self, split=None):
        """Resolve target field.

        Extract target field from `self._params`.
        """
        p = self._params

        if p is None:  # params is shallow
            return "sources"

        if split and hasattr(p, split):
            return getattr(getattr(p, split), "target_field")

        if hasattr(p, "target_field"):
            return p.target_field

        if hasattr(p, "complete"):
            return getattr(p.complete, "target_field")

        return "sources"

    def _canonicalize_initial_data(self):
        for split, container in {
            "train": self._train_data,
            "test": self._test_data,
            None: self._complete_data,
        }.items():
            if container is not None:
                self._canonicalize_container(
                    container, self._resolve_target_field(split)
                )

    def _canonicalize_container(self, container, target_field):
        """Canonicalize dataset fields.

        Map dataset-specific keys to canonical keys for downstream models.

        Note: Positions and masks are considered standard and are not remapped. SEDs are processed separately by the converter if needed.
        """
        # Lowercase canonical keys if they exist
        for key in self._canonical_keys:
            if key in container:
                continue  # already canonical
            for legacy_key in container.keys():
                if legacy_key.lower() == key.lower():
                    container[key] = container.pop(legacy_key)
                    break

        if target_field in container:
            # Set "sources" and drop "noisy_stars"
            container["sources"] = container.pop(target_field)
        else:
            raise KeyError(
                f"Target field '{target_field}' not found. "
                f"Available fields: {list(container.keys())}"
            )

        # Check if other canonical keys exist (positions, masks, seds)
        for key in self._canonical_keys:
            if key not in container:
                container[key] = None
                logger.warning(f"{key} not found in dataset.")

        return container

    @property
    def structure_state(self):
        """Return the current structural state of the dataset."""
        return self._structure_state

    @property
    def representation_state(self):
        """Return the current representation state of the dataset."""
        return self._representation_state

    @property
    def complete_data(self):
        """Return the complete dataset in the current representation."""
        if self._representation_state == RepresentationState.TENSORFLOW:
            return self._complete_tf or self._complete_data
        return self._complete_data

    @property
    def train_data(self):
        """Return the training set in the current representation."""
        if self._representation_state == RepresentationState.TENSORFLOW:
            return self._train_tf or self._train_data
        return self._train_data

    @property
    def test_data(self):
        """Return the test set in the current representation."""
        if self._representation_state == RepresentationState.TENSORFLOW:
            return self._test_tf or self._test_data
        return self._test_data

    # Read access to data params and metadata
    @property
    def params(self) -> Optional[Any]:
        """Get dataset params."""
        return self._params

    @property
    def metadata(self) -> Optional[dict]:
        """Get dataset metadata."""
        return self._metadata

    # Convenient access to canonical fields for downstream models
    @property
    def sources(self):
        """Get sources."""
        if self._complete_data is None:
            return None
        return self._complete_data.get("sources", None)

    @property
    def positions(self):
        """Get positions."""
        if self._complete_data is None:
            return None
        return self._complete_data.get("positions", None)

    @property
    def masks(self):
        """Get masks."""
        if self._complete_data is None:
            return None
        return self._complete_data.get("masks", None)

    @property
    def zernike_prior(self):
        """Get Zernike prior."""
        if self._complete_data is None:
            return None
        return self._complete_data.get("zernike_prior", None)

    def split_data(self, ratio: Optional[float] = None, seed: Optional[int] = None):
        """Split the complete dataset into train and test sets if not already split.

        Parameters
        ----------
        ratio : float, optional
            The fraction of the dataset to use for training (default is 0.8 or from params).
        seed : int, optional
            The random seed for reproducibility (default is from params).

        Raises
        ------
        RuntimeError
            If the dataset is not in COMPLETE state when attempting to split.
        """
        if self._structure_state != StructureState.COMPLETE:
            raise RuntimeError("Split only allowed from COMPLETE state.")

        # No need to split if initial loaded data was already split (idempotent)
        if self._train_data is None or self._test_data is None:
            ratio = ratio if ratio is not None else self._train_fraction
            logger.info(f"Setting train_fraction to {ratio}...")

            seed = seed if seed is not None else self._seed
            logger.info(f"Setting random seed for train-test split to {seed}....")

            self._train_data, self._test_data = self._split(
                self._complete_data, ratio=ratio, seed=seed
            )

        self._structure_state = StructureState.SPLIT

    def join_data(self, keys: Optional[List[str]] = None):
        """Join train/test dictionaries into complete dataset."""
        if self._structure_state != StructureState.SPLIT:
            raise RuntimeError("Join only allowed from SPLIT state.")

        if self._train_data is None or self._test_data is None:
            raise RuntimeError("Train or test data is missing; cannot join.")

        # Join on specified keys or CANONICAL_KEYS by default
        keys_to_join = keys or self._canonical_keys

        self._complete_data = {
            k: np.concatenate([self._train_data[k], self._test_data[k]], axis=0)
            for k in keys_to_join
            if k in self._train_data and k in self._test_data
        }

        self._structure_state = StructureState.COMPLETE

    def convert_to_tensorflow(self, simPSF, n_bins_lambda):
        """Convert to TensorFlow.

        Parameters
        ----------
        simPSF : PSFSimulator
            An instance of the PSFSimulator used to access the SED interpolator based on instrument specifications.

        n_bins_lambda : int
            Number of bins in wavelength for interpolation
        """
        if self._representation_state == RepresentationState.TENSORFLOW:
            return

        if self._converter is None:
            raise RuntimeError("No converter provided.")

        required_keys = tuple(self._canonical_keys)

        if self._structure_state == StructureState.SPLIT:
            self._train_tf = self._converter.convert_dataset(
                self._train_data, simPSF, n_bins_lambda, required_keys=required_keys
            )
            self._test_tf = self._converter.convert_dataset(
                self._test_data, simPSF, n_bins_lambda, required_keys=required_keys
            )
        else:
            self._complete_tf = self._converter.convert_dataset(
                self._complete_data, simPSF, n_bins_lambda, required_keys=required_keys
            )

        self._representation_state = RepresentationState.TENSORFLOW

    def _split(self, data, ratio: Optional[float] = None, seed: Optional[int] = None):
        """Split the data into train and test sets based on the specified ratio and seed.

        Parameters
        ----------
        data : DatasetContainer
            Container holding the complete dataset to be split. The container
            behaves like a dictionary and stores arrays for different dataset
            components (e.g., sources, positions, masks, SEDs), with optional
            attribute-style access. Only array-like entries whose first dimension
            corresponds to the number of samples will be split.
        ratio : float, optional
            Fraction of the dataset to allocate to the training set.
        seed : int, optional
            Random seed used to generate the split for reproducibility.

        """
        ratio = ratio or getattr(self._params, "train_fraction", 0.8)
        rng = np.random.default_rng(seed)

        canonical_keys = self._canonical_keys

        n = None

        # Determine sample size from canonical keys
        for key in canonical_keys:
            if key in data and isinstance(data[key], np.ndarray):
                n = data[key].shape[0]
                break

        if n is None:
            raise ValueError(
                f"Could not determine dataset length from canonical keys {canonical_keys}"
            )

        n_train = int(n * ratio)
        indices = rng.permutation(n)

        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_data = {}
        test_data = {}

        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.shape[0] == n:
                train_data[k] = v[train_idx]
                test_data[k] = v[test_idx]
            else:
                # leave arrays with different leading axis untouched
                train_data[k] = v
                test_data[k] = v

        return train_data, test_data

    def release_numpy(self):
        """Release NumPy datasets."""
        if self._representation_state != RepresentationState.TENSORFLOW:
            raise RuntimeError("NumPy can only be released after TF conversion.")

        self._complete_data = DatasetContainer({})
        self._train_data = DatasetContainer({})
        self._test_data = DatasetContainer({})

    def release_tensorflow(self):
        """Release tensorflow datasets."""
        self._complete_tf = None
        self._train_tf = None
        self._test_tf = None

        self._representation_state = RepresentationState.NUMPY
