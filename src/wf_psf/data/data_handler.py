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
import logging

logger = logging.getLogger(__name__)


class DataHandler:
    """Data Handler.

    This class manages loading and processing of training and testing data for use during PSF model training and validation.
    It provides methods to access and preprocess the data.

    Parameters
    ----------
    dataset_type: str
        A string indicating type of data ("train" or "test").
    data_params: Recursive Namespace object
        Recursive Namespace object containing training data parameters
    simPSF: PSFSimulator
        An instance of the PSFSimulator class for simulating a PSF.
    n_bins_lambda: int
        The number of bins in wavelength.
    load_data: bool, optional
        A flag used to control data loading steps. If True, data is loaded and processed
        during initialization. If False, data loading is deferred until explicitly called.

    Attributes
    ----------
    dataset_type: str
        A string indicating the type of dataset ("train" or "test").
    data_params: Recursive Namespace object
        A Recursive Namespace object containing training or test data parameters.
    dataset: dict
        A dictionary containing the loaded dataset, including positions and stars/noisy_stars.
    simPSF: object
        An instance of the SimPSFToolkit class for simulating PSF.
    n_bins_lambda: int
        The number of bins in wavelength.
    sed_data: tf.Tensor
        A TensorFlow tensor containing the SED data for training/testing.
    load_data_on_init: bool, optional
        A flag used to control data loading steps. If True, data is loaded and processed
        during initialization. If False, data loading is deferred until explicitly called.
    """

    def __init__(self, dataset_type, data_params, simPSF, n_bins_lambda, load_data: bool=True):
        """
        Initialize the dataset handler for PSF simulation.

        Parameters
        ----------
        dataset_type : str
            A string indicating the type of data ("train" or "test").
        data_params : Recursive Namespace object
            A Recursive Namespace object containing parameters for both 'train' and 'test' datasets.
        simPSF : PSFSimulator
            An instance of the PSFSimulator class for simulating a PSF.
        n_bins_lambda : int
            The number of bins in wavelength.
        load_data : bool, optional
            A flag to control whether data should be loaded and processed during initialization.
            If True, data is loaded and processed during initialization; if False, data loading
            is deferred until explicitly called.
        """
        self.dataset_type = dataset_type
        self.data_params = data_params.__dict__[dataset_type]
        self.simPSF = simPSF
        self.n_bins_lambda = n_bins_lambda
        self.load_data_on_init = load_data
        if self.load_data_on_init:
            self.load_dataset()
            self.process_sed_data()
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
            else:
                logger.warning(f"Missing 'stars' in {self.dataset_type} dataset.")
        elif "inference" == self.dataset_type:
            pass

    def process_sed_data(self):
        """Process SED Data.

        A method to generate and process SED data.

        """
        self.sed_data = [
            utils.generate_SED_elems_in_tensorflow(
                _sed, self.simPSF, n_bins=self.n_bins_lambda, tf_dtype=tf.float64
            )
            for _sed in self.dataset["SEDs"]
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
        key for key, dataset in [(train_key, data.training_data.dataset), (test_key, data.test_data.dataset)]
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

