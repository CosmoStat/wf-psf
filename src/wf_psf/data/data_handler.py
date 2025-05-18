"""Data Handler Module.

Provides tools for loading, preprocessing, and managing data used in both training and inference workflows.

Includes:
- The `DataHandler` class for managing datasets and associated metadata
- Utility functions for loading structured data products
- Preprocessing routines for spectral energy distributions (SEDs), including format conversion (e.g., to TensorFlow) and transformations

This module serves as a central interface between raw data and modeling components.

Authors: Jennifer Pollack <jennifer.pollack@cea.fr>, Tobias Liaudat <tobiasliaudat@gmail.com>
"""

import os
import numpy as np
import wf_psf.utils.utils as utils
from wf_psf.psf_models.tf_modules.tf_utils import ensure_tensor
import tensorflow as tf
from typing import Optional, Union
import numpy as np
import tensorflow as tf
from fractions import Fraction
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class DataHandler:
    """
    DataHandler for WaveDiff PSF modeling.

    This class manages loading, preprocessing, and TensorFlow conversion of datasets used
    for PSF model training, testing, and inference in the WaveDiff framework.

    Parameters
    ----------
    dataset_type : str
        Indicates the dataset mode ("train", "test", or "inference").
    data_params : RecursiveNamespace
        Configuration object containing dataset parameters (e.g., file paths, preprocessing flags).
    simPSF : PSFSimulator
        An instance of the PSFSimulator class used to encode SEDs into a TensorFlow-compatible format.
    n_bins_lambda : int
        Number of wavelength bins used to discretize SEDs.
    load_data : bool, optional
        If True (default), loads and processes data during initialization. If False, data loading
        must be triggered explicitly.
    dataset : dict or list, optional
        If provided, uses this pre-loaded dataset instead of triggering automatic loading.
    sed_data : dict or list, optional
        If provided, uses this SED data directly instead of extracting it from the dataset.

    Attributes
    ----------
    dataset_type : str
        Indicates the dataset mode ("train", "test", or "inference").
    data_params : RecursiveNamespace
        Configuration parameters for data access and structure.
    simPSF : PSFSimulator
        Simulator used to transform SEDs into TensorFlow-ready tensors.
    n_bins_lambda : int
        Number of wavelength bins in the SED representation.
    load_data_on_init : bool
        Whether data was loaded automatically during initialization.
    dataset : dict
        Loaded dataset including keys such as 'positions', 'stars', 'noisy_stars', or similar.
    sed_data : tf.Tensor
        TensorFlow-formatted SED data with shape [batch_size, n_bins_lambda, features].
    """

    def __init__(
        self,
        dataset_type,
        data_params,
        simPSF,
        n_bins_lambda,
        load_data: bool = True,
        dataset: Optional[Union[dict, list]] = None,
        sed_data: Optional[Union[dict, list]] = None,
    ):
        """
                Initialize the DataHandler for PSF dataset preparation.

                This constructor sets up the dataset handler used for PSF simulation tasks,
                such as training, testing, or inference. It supports three modes of use:

                1. **Manual mode** (`load_data=False`, no `dataset`): data loading and SED processing
                   must be triggered manually via `load_dataset()` and `process_sed_data()`.
                2. **Pre-loaded dataset mode** (`dataset` is provided): the given dataset is used directly,
                   and `process_sed_data()` is called with either the given `sed_data` or `dataset["SEDs"]`.
                3. **Automatic loading mode** (`load_data=True` and no `dataset`): the dataset is loaded
                   from disk using `data_params`, and SEDs are extracted and processed automatically.

                Parameters
                ----------
                dataset_type : str
                    One of {"train", "test", "inference"} indicating dataset usage.
                data_params : RecursiveNamespace
                    Configuration object with paths, preprocessing options, and metadata.
                simPSF : PSFSimulator
                    Used to convert SEDs to TensorFlow format.
                n_bins_lambda : int
                    Number of wavelength bins for the SEDs.
                load_data : bool, optional
                    Whether to automatically load and process the dataset (default: True).
                dataset : dict or list, optional
                    A pre-loaded dataset to use directly (overrides `load_data`).
                sed_data : array-like, optional
        <<<<<<< HEAD
                    Pre-loaded SED data to use directly. If not provided but `dataset` is,
        =======
                    Pre-loaded SED data to use directly. If not provided but `dataset` is,
        >>>>>>> 4b896e3 (Refactor data_handler with new utility functions to validate and process datasets and update docstrings)
                    SEDs are taken from `dataset["SEDs"]`.

                Raises
                ------
                ValueError
                    If SEDs cannot be found in either `dataset` or as `sed_data`.

                Notes
                -----
                - `self.dataset` and `self.sed_data` are both `None` if neither `dataset` nor
                  `load_data=True` is used.
                - TensorFlow conversion is performed at the end of initialization via `convert_dataset_to_tensorflow()`.
        """

        self.dataset_type = dataset_type
        self.data_params = data_params
        self.simPSF = simPSF
        self.n_bins_lambda = n_bins_lambda
        self.load_data_on_init = load_data

        if dataset is not None:
            self.dataset = dataset
            self.process_sed_data(sed_data)
            self.validate_and_process_dataset()
        elif self.load_data_on_init:
            self.load_dataset()
            self.process_sed_data(self.dataset["SEDs"])
            self.validate_and_process_dataset()
        else:
            self.dataset = None
            self.sed_data = None

    @property
    def tf_positions(self):
        return ensure_tensor(self.dataset["positions"])

    def load_dataset(self):
        """Load dataset.

        Load the dataset based on the specified dataset type.

        """
        self.dataset = np.load(
            os.path.join(self.data_params.data_dir, self.data_params.file),
            allow_pickle=True,
        )[()]

    def validate_and_process_dataset(self):
        """Validate the dataset structure and convert fields to TensorFlow tensors."""
        self._validate_dataset_structure()
        self._convert_dataset_to_tensorflow()

    def _validate_dataset_structure(self):
        """Validate dataset structure based on dataset_type."""
        if self.dataset is None:
            raise ValueError("Dataset is None")

        if "positions" not in self.dataset:
            raise ValueError("Dataset missing required field: 'positions'")

        if self.dataset_type == "training":
            if "noisy_stars" not in self.dataset:
                raise ValueError(
                    f"Missing required field 'noisy_stars' in {self.dataset_type} dataset."
                )
        elif self.dataset_type == "test":
            if "stars" not in self.dataset:
                raise ValueError(
                    f"Missing required field 'stars' in {self.dataset_type} dataset."
                )
        elif self.dataset_type == "inference":
            pass
        else:
            raise ValueError(f"Unrecognized dataset_type: {self.dataset_type}")

    def _convert_dataset_to_tensorflow(self):
        """Convert dataset to TensorFlow tensors."""
        self.dataset["positions"] = ensure_tensor(
            self.dataset["positions"], dtype=tf.float32
        )

        if self.dataset_type == "train":
            self.dataset["noisy_stars"] = ensure_tensor(
                self.dataset["noisy_stars"], dtype=tf.float32
            )
        elif self.dataset_type == "test":
            self.dataset["stars"] = ensure_tensor(
                self.dataset["stars"], dtype=tf.float32
            )

    def process_sed_data(self, sed_data):
        """
        Generate and process SED (Spectral Energy Distribution) data.

        This method transforms raw SED inputs into TensorFlow tensors suitable for model input.
        It generates wavelength-binned SED elements using the PSF simulator, converts the result
        into a tensor, and transposes it to match the expected shape for training or inference.

        Parameters
        ----------
        sed_data : list or array-like
            A list or array of raw SEDs, where each SED is typically a vector of flux values
            or coefficients. These will be processed using the PSF simulator.

        Raises
        ------
        ValueError
            If `sed_data` is None.

        Notes
        -----
        The resulting tensor is stored in `self.sed_data` and has shape
        `(num_samples, n_bins_lambda, n_components)`, where:
            - `num_samples` is the number of SEDs,
            - `n_bins_lambda` is the number of wavelength bins,
            - `n_components` is the number of components per SED (e.g., filters or basis terms).

        The intermediate tensor is created with `tf.float64` for precision during generation,
        but is converted to `tf.float32` after processing for use in training.
        """
        if sed_data is None:
            raise ValueError("SED data must be provided explicitly or via dataset.")

        self.sed_data = [
            utils.generate_SED_elems_in_tensorflow(
                _sed, self.simPSF, n_bins=self.n_bins_lambda, tf_dtype=tf.float64
            )
            for _sed in sed_data
        ]
        # Convert list of generated SED tensors to a single TensorFlow tensor of float32 dtype
        self.sed_data = ensure_tensor(self.sed_data, dtype=tf.float32)
        self.sed_data = tf.transpose(self.sed_data, perm=[0, 2, 1])


def extract_star_data(data, train_key: str, test_key: str) -> np.ndarray:
    """
    Extract and concatenate star-related data from training and test datasets.

    This function retrieves arrays (e.g., postage stamps, masks, positions) from
    both the training and test datasets using the specified keys, converts them
    to NumPy if necessary, and concatenates them along the first axis.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.
    train_key : str
        Key to retrieve data from the training dataset
        (e.g., 'noisy_stars', 'masks').
    test_key : str
        Key to retrieve data from the test dataset
        (e.g., 'stars', 'masks').

    Returns
    -------
    np.ndarray
        Concatenated NumPy array containing the selected data from both
        training and test sets.

    Raises
    ------
    KeyError
        If either the training or test dataset does not contain the
        requested key.

    Notes
    -----
    - Designed for datasets with separate train/test splits, such as when
      evaluating metrics on held-out data.
    - TensorFlow tensors are automatically converted to NumPy arrays.
    - Requires eager execution if TensorFlow tensors are present.
    """
    # Ensure the requested keys exist in both training and test datasets
    missing_keys = [
        key
        for key, dataset in [
            (train_key, data.training_data.dataset),
            (test_key, data.test_data.dataset),
        ]
        if key not in dataset
    ]

    if missing_keys:
        raise KeyError(f"Missing keys in dataset: {missing_keys}")

    # Retrieve data from training and test sets
    train_data = data.training_data.dataset[train_key]
    test_data = data.test_data.dataset[test_key]

    # Convert to NumPy if necessary
    if tf.is_tensor(train_data):
        train_data = train_data.numpy()
    if tf.is_tensor(test_data):
        test_data = test_data.numpy()

    # Concatenate and return
    return np.concatenate((train_data, test_data), axis=0)


def get_data_array(
    data,
    run_type: str,
    key: str = None,
    train_key: str = None,
    test_key: str = None,
    allow_missing: bool = True,
) -> Optional[np.ndarray]:
    """
    Retrieve data from dataset depending on run type.

    This function provides a unified interface for accessing data across different
    execution contexts (training, simulation, metrics, inference). It handles
    key resolution with sensible fallbacks and optional missing data tolerance.

    Parameters
    ----------
    data : DataConfigHandler
        Dataset object containing training, test, or inference data.
        Expected to have methods compatible with the specified run_type.
    run_type : {"training", "simulation", "metrics", "inference"}
        Execution context that determines how data is retrieved:
        - "training", "simulation", "metrics": Uses extract_star_data function
        - "inference": Retrieves data directly from dataset using key lookup
    key : str, optional
        Primary key for data lookup. Used directly for inference run_type.
        If None, falls back to train_key value. Default is None.
    train_key : str, optional
        Key for training dataset access. If None and key is provided,
        defaults to key value. Default is None.
    test_key : str, optional
        Key for test dataset access. If None, defaults to the resolved
        train_key value. Default is None.
    allow_missing : bool, default True
        Control behavior when data is missing or keys are not found:
        - True: Return None instead of raising exceptions
        - False: Raise appropriate exceptions (KeyError, ValueError)

    Returns
    -------
    np.ndarray or None
        Retrieved data as NumPy array. Returns None only when allow_missing=True
        and the requested data is not available.

    Raises
    ------
    ValueError
        If run_type is not one of the supported values, or if no key can be
        resolved for the operation and allow_missing=False.
    KeyError
        If the specified key is not found in the dataset and allow_missing=False.

    Notes
    -----
    Key resolution follows this priority order:
    1. train_key = train_key or key
    2. test_key = test_key or resolved_train_key
    3. key = key or resolved_train_key (for inference fallback)

    For TensorFlow tensors, the .numpy() method is called to convert to NumPy.
    Other data types are converted using np.asarray().

    Examples
    --------
    >>> # Training data retrieval
    >>> train_data = get_data_array(data, "training", train_key="noisy_stars")

    >>> # Inference with fallback handling
    >>> inference_data = get_data_array(data, "inference", key="positions",
    ...                                allow_missing=True)
    >>> if inference_data is None:
    ...     print("No inference data available")

    >>> # Using key parameter for both train and inference
    >>> result = get_data_array(data, "inference", key="positions")
    """
    # Validate run_type early
    valid_run_types = {"training", "simulation", "metrics", "inference"}
    if run_type not in valid_run_types:
        raise ValueError(f"run_type must be one of {valid_run_types}, got '{run_type}'")

    # Simplify key resolution with clear precedence
    effective_train_key = train_key or key
    effective_test_key = test_key or effective_train_key
    effective_key = key or effective_train_key

    try:
        if run_type in {"simulation", "training", "metrics"}:
            return extract_star_data(data, effective_train_key, effective_test_key)
        else:  # inference
            return _get_direct_data(data, effective_key, allow_missing)
    except Exception:
        if allow_missing:
            return None
        raise


def _get_direct_data(data, key: str, allow_missing: bool) -> Optional[np.ndarray]:
    """
    Extract data directly with proper error handling and type conversion.

    Parameters
    ----------
    data : DataConfigHandler
        Dataset object with a .dataset attribute that supports .get() method.
    key : str or None
        Key to lookup in the dataset. If None, behavior depends on allow_missing.
    allow_missing : bool
        If True, return None for missing keys/data instead of raising exceptions.

    Returns
    -------
    np.ndarray or None
        Data converted to NumPy array, or None if allow_missing=True and
        data is unavailable.

    Raises
    ------
    ValueError
        If key is None and allow_missing=False.
    KeyError
        If key is not found in dataset and allow_missing=False.

    Notes
    -----
    Conversion logic:
    - TensorFlow tensors: Converted using .numpy() method
    - Other types: Converted using np.asarray()
    """
    if key is None:
        if allow_missing:
            return None
        raise ValueError("No key provided for inference data")

    value = data.dataset.get(key, None)
    if value is None:
        if allow_missing:
            return None
        raise KeyError(f"Key '{key}' not found in inference dataset")

    return value.numpy() if tf.is_tensor(value) else np.asarray(value)
