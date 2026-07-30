"""
Data Adapter.

This module manages dataset lifecycle transitions for the WF-PSF pipeline.

Overview
--------
Two orthogonal state machines are maintained:

**1. Structure state**

- ``COMPLETE`` → ``SPLIT`` via :meth:`split_data`
- ``SPLIT`` → ``COMPLETE`` via :meth:`join_data`

**2. Representation state**

- ``NUMPY`` → ``TENSORFLOW`` via :meth:`convert_to_tensorflow`


Glossary
--------
**COMPLETE**
    Dataset stored as a single container.

**SPLIT**
    Dataset stored as train/test subsets.

**NUMPY**
    Data stored as NumPy arrays.

**TENSORFLOW**
    Data stored as TensorFlow tensors.


Design principles
-----------------
- Structure and representation are orthogonal.
- All transitions are explicit and idempotent where possible.
- No training or model logic lives in this module.
- Dataset field names are canonicalized for downstream models.

Notes
-----
The :class:`DataAdapter` class manages these transitions while providing
a consistent interface for accessing dataset contents.

Authors: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from enum import Enum, auto
import numpy as np
from typing import Any, Optional
from wf_psf.data.constants import (
    CANONICAL_DATASET_KEYS,
    DEFAULT_SEED,
    DEFAULT_TRAIN_FRACTION,
    DATASET_INDEX_KEY,
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
        self._canonical_keys = getattr(params, "canonical_keys", CANONICAL_DATASET_KEYS)
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
        """Initialize internal data containers based on structure state.

        Copies input dataset into internal container representation,
        handling either COMPLETE or SPLIT structure modes.
        """
        if self._structure_state == StructureState.COMPLETE:
            self._complete_data = to_container(dataset.complete.copy())
            return

        self._train_data = to_container(dataset.train.copy())
        self._test_data = to_container(dataset.test.copy())

    def _resolve_target_field(self, split=None):
        """Resolve the target field name from adapter parameters.

        Resolution order:
        1. Split-specific field (if `split` provided)
        2. Global `target_field`
        3. `complete.target_field`
        4. Default to "sources"

        Returns
        -------
        str
            Name of the field to use as the source/target data.
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
        """Canonicalize all available dataset splits.

        Applies field normalization to train, test, and complete containers
        using the resolved target field for each split.
        """
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
        """Canonicalize dataset fields in-place.

        Maps dataset-specific keys to canonical naming convention.

        Notes
        -----
        - Positions and masks are assumed to be standard and are not remapped.
        - Missing canonical fields are set to None with a warning.
        """
        # Rename keys (case-insensitive mapping)
        for key in self._canonical_keys:
            if key in container:
                continue  # already canonical
            for legacy_key in container.keys():
                if legacy_key.lower() == key.lower():
                    container[key] = container.pop(legacy_key)
                    break

        # Map target_field to sources
        if target_field in container:
            container["sources"] = container.pop(target_field)
        else:
            logger.warning(
                f"Target field '{target_field}' not found. "
                f"Available fields: {list(container.keys())}"
            )

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
        """Get sources for the complete dataset."""
        if self._complete_data is None:
            return None
        return self._complete_data.get("sources", None)

    @property
    def positions(self):
        """Get positions for the complete dataset."""
        if self._complete_data is None:
            return None
        return self._complete_data.get("positions", None)

    @property
    def masks(self):
        """Get masks for the complete dataset."""
        if self._complete_data is None:
            return None
        return self._complete_data.get("masks", None)

    @property
    def zernike_prior(self):
        """Get Zernike prior for the complete dataset."""
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

        Notes
        -----
        - Splitting is idempotent: if the dataset is already in SPLIT state,
          this method does not modify the data or re-split the dataset.
        """
        if self._structure_state != StructureState.COMPLETE:
            raise RuntimeError("Split only allowed from COMPLETE state.")

        # -------------------------------
        # Consistency enforcement
        # -------------------------------
        if (self._train_data is None) != (self._test_data is None):
            raise RuntimeError("Inconsistent split state: train/test mismatch.")

        # Idempotent behaviour
        if self._train_data is None and self._test_data is None:
            ratio = ratio if ratio is not None else self._train_fraction
            logger.info(f"Setting train_fraction to {ratio}...")

            seed = seed if seed is not None else self._seed
            logger.info(f"Setting random seed for train-test split to {seed}....")

            self._train_data, self._test_data = self._split(
                self._complete_data, ratio=ratio, seed=seed
            )
        else:
            logger.info(
                "Split data already exists; reusing existing train/test split (idempotent call)."
            )
        self._structure_state = StructureState.SPLIT

    def join_data(self, keys: Optional[list[str]] = None):
        """Join train and test splits into a single complete dataset.

        Concatenates corresponding arrays from the train and test containers
        along the first axis (sample dimension) for the specified keys.

        Parameters
        ----------
        keys : list of str, optional
            Dataset fields to join. If None, uses the canonical dataset keys.

        Raises
        ------
        RuntimeError
            If the adapter is not in SPLIT state or if train/test data is missing.

        Notes
        -----
        Only keys present in both train and test datasets are joined.
        """
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

    def convert_to_tensorflow(self, simPSF, n_bins_lambda, mode):
        """Convert dataset containers from NumPy to TensorFlow representation.

        Applies the configured converter to transform dataset fields associated
        with canonical keys into TensorFlow-compatible formats.

        Conversion is performed on the active structure:

        - SPLIT: converts train and test datasets separately
        - COMPLETE: converts the full dataset

        Parameters
        ----------
        simPSF : PSFSimulator
            Simulator instance passed to the converter.
        n_bins_lambda : int
            Number of wavelength bins used during conversion.
        mode : DatasetMode
            Dataset operation mode used to select the appropriate dataset schema for a given pipeline process (e.g. training, validation, inference) 

        Raises
        ------
        RuntimeError
            If no converter is configured.

        Notes
        -----
        - Conversion is idempotent: if the data is already in TensorFlow
        representation, this method does nothing.

        - Converted datasets are stored in internal attributes
        (``_train_tf``, ``_test_tf``, ``_complete_tf``) and do not overwrite
        the original NumPy data.
        """
        if self._representation_state == RepresentationState.TENSORFLOW:
            return

        if self._converter is None:
            raise RuntimeError("No converter provided.")

        if self._structure_state == StructureState.SPLIT:

            n_train = len(self._train_data[DATASET_INDEX_KEY])
            n_test = len(self._test_data[DATASET_INDEX_KEY])
    
            logger.info(
                f"Converting training dataset to TensorFlow "
                f"(mode={mode.name.lower()}, samples={n_train})..."
            )

            self._train_tf = self._converter.convert_dataset(
                self._train_data, simPSF, n_bins_lambda, mode=mode
            )

            logger.info(
                f"Converting test dataset to TensorFlow "
                f"(mode={mode.name.lower()}, samples={n_test})..."
            )

            self._test_tf = self._converter.convert_dataset(
                self._test_data, simPSF, n_bins_lambda, mode=mode
            )
        else:
            n_complete = len(self._complete_data[DATASET_INDEX_KEY])
            
            logger.info(
                f"Converting complete dataset to TensorFlow "
                f"(mode={mode.name.lower()}, samples={n_complete})..."
            )

            self._complete_tf = self._converter.convert_dataset(
                self._complete_data, simPSF, n_bins_lambda, mode=mode
            )

        self._representation_state = RepresentationState.TENSORFLOW
        logger.info("Dataset representation state updated to TensorFlow.")

    def _split(self, data, ratio: Optional[float] = None, seed: Optional[int] = None):
        """Split a dataset container into train and test subsets.

        The split is performed along the first dimension (sample axis) using a
        random permutation. Only array entries whose leading dimension matches
        the number of samples are split; all other entries are copied unchanged.

        Parameters
        ----------
        data : DatasetContainer
            Container holding the complete dataset. Expected to store array-like
            values (e.g. sources, positions, masks, SEDs) indexed by field name.
        ratio : float, optional
            Fraction of samples assigned to the training set. If not provided,
            defaults to ``self._params.train_fraction`` or 0.8.
        seed : int, optional
            Random seed used to generate the split.

        Returns
        -------
        train_data : dict
            Dictionary containing the training subset.
        test_data : dict
            Dictionary containing the test subset.

        Raises
        ------
        ValueError
            If the dataset size cannot be inferred from canonical keys.

        Notes
        -----
        - The dataset size is inferred from the first available canonical key.
        - Arrays whose leading dimension does not match the inferred size are
        not split and are copied as-is into both outputs.
        """
        ratio = ratio or getattr(self._params, "train_fraction", 0.8)
        rng = np.random.default_rng(seed)

        canonical_keys = self._canonical_keys

        n = None

        # Determine sample size from index dataset key
        n = data[DATASET_INDEX_KEY].shape[0]

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

    
