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
            Pre-loaded SED data to use directly. If not provided but `dataset` is,
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

        if self.dataset_type == "train":
            if "noisy_stars" not in self.dataset:
                logger.warning("Missing 'noisy_stars' in 'train' dataset.")
        elif self.dataset_type == "test":
            if "stars" not in self.dataset:
                logger.warning("Missing 'stars' in 'test' dataset.")
        elif self.dataset_type == "inference":
            pass
        else:
            logger.warning(f"Unrecognized dataset_type: {self.dataset_type}")

    def _convert_dataset_to_tensorflow(self):
        """Convert dataset to TensorFlow tensors."""

        self.dataset["positions"] = tf.convert_to_tensor(
            self.dataset["positions"], dtype=tf.float32
        )
        if self.dataset_type == "training":
            if "noisy_stars" in self.dataset:
                self.dataset["noisy_stars"] = tf.convert_to_tensor(
                    self.dataset["noisy_stars"], dtype=tf.float32
                )
            else:
                logger.warning(f"Missing 'noisy_stars' in {self.dataset_type} dataset.")
        elif self.dataset_type == "test":
            if "stars" in self.dataset:
                self.dataset["stars"] = tf.convert_to_tensor(
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
        self.sed_data = tf.convert_to_tensor(self.sed_data, dtype=tf.float32)
        self.sed_data = tf.transpose(self.sed_data, perm=[0, 2, 1])


def get_np_obs_positions(data):
    """Get observed positions in numpy from the provided dataset.

    This method concatenates the positions of the stars from both the training
    and test datasets to obtain the observed positions.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    np.ndarray
        Numpy array containing the observed positions of the stars.

    Notes
    -----
    The observed positions are obtained by concatenating the positions of stars
    from both the training and test datasets along the 0th axis.
    """
    obs_positions = np.concatenate(
        (
            data.training_data.dataset["positions"],
            data.test_data.dataset["positions"],
        ),
        axis=0,
    )

    return obs_positions


def get_obs_positions(data):
    """Get observed positions from the provided dataset.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    tf.Tensor
        Tensor containing the observed positions of the stars.
    """
    obs_positions = get_np_obs_positions(data)

    return tf.convert_to_tensor(obs_positions, dtype=tf.float32)


def extract_star_data(data, train_key: str, test_key: str) -> np.ndarray:
    """Extract specific star-related data from training and test datasets.

    This function retrieves and concatenates specific star-related data (e.g., stars, masks) from the
    star training and test datasets such as star stamps or masks, based on the provided keys.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.
    train_key : str
        The key to retrieve data from the training dataset (e.g., 'noisy_stars', 'masks').
    test_key : str
        The key to retrieve data from the test dataset (e.g., 'stars', 'masks').

    Returns
    -------
    np.ndarray
        A NumPy array containing the concatenated data for the given keys.

    Raises
    ------
    KeyError
        If the specified keys do not exist in the training or test datasets.

    Notes
    -----
    - If the dataset contains TensorFlow tensors, they will be converted to NumPy arrays.
    - Ensure that eager execution is enabled when calling this function.
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
