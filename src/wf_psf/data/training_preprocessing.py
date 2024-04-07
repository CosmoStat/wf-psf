"""Training Data Processing.

A module to load and preprocess training and validation test data.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr> and Tobias Liaudat <tobiasliaudat@gmail.com>

"""

import numpy as np
import wf_psf.utils.utils as utils
import tensorflow as tf
import os


class DataHandler:
    """Data Handler.

    This class manages loading and processing of training and testing data for use in machine learning models.
    It provides methods to access and preprocess the data.

    Parameters
    ----------
    data_type: str
        A string indicating type of data ("train" or "test").
    data_params: Recursive Namespace object
        Recursive Namespace object containing training data parameters
    simPSF: PSFSimulator
        An instance of the PSFSimulator class for simulating a PSF.
    n_bins_lambda: int
        The number of bins in wavelength.
    init_flag: bool, optional
        A flag indicating whether to perform initialization steps upon object creation.
        If True (default), the dataset is loaded and processed. If False, initialization
        steps are skipped, and manual initialization is required.

    Attributes
    ----------
    dataset: dict
        A dictionary containing the loaded dataset, including positions and stars/noisy_stars.
    simPSF: object
        An instance of the SimPSFToolkit class for simulating PSF.
    n_bins_lambda: int
        The number of bins in wavelength.
    sed_data: tf.Tensor
        A TensorFlow tensor containing the SED data for training/testing.
    init_flag: bool, optional
        A flag used to control initialization steps. If True, initialization is performed
        upon object creation.


    """

    def __init__(self, data_type, data_params, simPSF, n_bins_lambda, init_flag=True):
        self.data_params = data_params.__dict__[data_type]
        self.simPSF = simPSF
        self.n_bins_lambda = n_bins_lambda
        self.dataset = None
        self.sed_data = None
        self.initialize(init_flag)

    def initialize(self, init_flag):
        """Initialize.

        Initialize the DataHandler instance by loading and processing the dataset,
        if the init_flag is True.

        Parameters
        ----------
        init_flag : bool
            A flag indicating whether to perform initialization steps. If True,
            the dataset is loaded and processed. If False, initialization steps
            are skipped.

        """
        if init_flag:
            self.load_dataset()
            self.process_sed_data()

    def load_dataset(self):
        """Load dataset.

        Load the dataset based on the specified data type.

        """
        self.dataset = np.load(
            os.path.join(self.data_params.data_dir, self.data_params.file),
            allow_pickle=True,
        )[()]
        self.dataset["positions"] = tf.convert_to_tensor(
            self.dataset["positions"], dtype=tf.float32
        )
        if "train" in self.data_params.file:
            self.dataset["noisy_stars"] = tf.convert_to_tensor(
                self.dataset["noisy_stars"], dtype=tf.float32
            )
        elif "test" in self.data_params.file:
            self.dataset["stars"] = tf.convert_to_tensor(
                self.dataset["stars"], dtype=tf.float32
            )

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


def get_obs_positions(data):
    """Get observed positions from the provided dataset.

    This method concatenates the positions of the stars from both the training
    and test datasets to obtain the observed positions.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    tf.Tensor
        Tensor containing the observed positions of the stars.

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
    return tf.convert_to_tensor(obs_positions, dtype=tf.float32)


def get_zernike_prior(model_params, data):
    """Get Zernike priors from the provided dataset.

    This method concatenates the Zernike priors from both the training
    and test datasets.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Object containing parametrs for this PSF model class.
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    tf.Tensor
        Tensor containing the observed positions of the stars.

    Notes
    -----
    The Zernike prior are obtained by concatenating the Zernike priors
    from both the training and test datasets along the 0th axis.

    """

    # Add new input variable (model_params), 
    # in order to have the different otions:
    # 1. prior / 2. centroids / 3. ccd_missalignments

    zernike_prior = np.concatenate(
        (
            data.training_data.dataset["zernike_prior"],
            data.test_data.dataset["zernike_prior"],
        ),
        axis=0,
    )
    return tf.convert_to_tensor(zernike_prior, dtype=tf.float32)
