"""Training Data Processing.

A module to load and preprocess training and validation test data.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr> and Tobias Liaudat <tobiasliaudat@gmail.com>

"""

import os
import numpy as np
import wf_psf.utils.utils as utils
import tensorflow as tf
from wf_psf.utils.ccd_misalignments import CCDMisalignmentCalculator
from wf_psf.utils.centroids import compute_zernike_tip_tilt
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

    def __init__(self, dataset_type, data_params, simPSF, n_bins_lambda, load_data=True):
        """
        Initialize the dataset handler for PSF simulation.

        Parameters
        ----------
        dataset_type: str
            A string indicating the type of data ("train" or "test").
        data_params: Recursive Namespace object
            A Recursive Namespace object containing parameters for both 'train' and 'test' datasets.
        simPSF: PSFSimulator
            An instance of the PSFSimulator class for simulating a PSF.
        n_bins_lambda: int
            The number of bins in wavelength.
        load_data: bool, optional
            A flag to control whether data should be loaded and processed during initialization.
            If True, data is loaded and processed during initialization; if False, data loading
            is deferred until explicitly called.
        """
        self.dataset_type = dataset_type
        self.data_params = data_params.__dict__[dataset_type]
        self.simPSF = simPSF
        self.n_bins_lambda = n_bins_lambda
        self.dataset = None
        self.sed_data = None
        self.load_data_on_init = load_data
        if self.load_data_on_init:
            self.load_dataset()
            self.process_sed_data()


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
        if "train" == self.dataset_type:
            if "noisy_stars" in self.dataset:
                self.dataset["noisy_stars"] = tf.convert_to_tensor(
                    self.dataset["noisy_stars"], dtype=tf.float32
                )
            else:
                logger.warning(f"Missing 'noisy_stars' in {self.dataset_type} dataset.")
        elif "test" == self.dataset_type:
            if "stars" in self.dataset:
                self.dataset["stars"] = tf.convert_to_tensor(
                    self.dataset["stars"], dtype=tf.float32
                )
            else:
                logger.warning(f"Missing 'stars' in {self.dataset_type} dataset.")


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


def get_np_zernike_prior(data):
    """Get the zernike prior from the provided dataset.

    This method concatenates the stars from both the training
    and test datasets to obtain the full prior.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    zernike_prior : np.ndarray
        Numpy array containing the full prior.
    """
    zernike_prior = np.concatenate(
        (
            data.training_data.dataset["zernike_prior"],
            data.test_data.dataset["zernike_prior"],
        ),
        axis=0,
    )

    return zernike_prior


def compute_centroid_correction(model_params, data) -> np.ndarray:
    """Compute centroid corrections using Zernike polynomials.

    This function calculates the Zernike contributions required to match the centroid
    of the WaveDiff PSF model to the observed star centroids.


    Parameters
    ----------
    model_params : RecursiveNamespace
        An object containing parameters for the PSF model, including pixel sampling
        and initial centroid window parameters.
    data : DataConfigHandler
        An object containing training and test datasets, including observed PSFs
        and optional star masks.

    Returns
    -------
    zernike_centroid_array : np.ndarray
         A 2D NumPy array of shape `(n_stars, 3)`, where `n_stars` is the number of 
        observed stars. The array contains the computed Zernike contributions, 
        with zero padding applied to the first column to ensure a consistent shape.
    """
    star_postage_stamps = extract_star_data(data=data, train_key="noisy_stars", test_key="stars")

    # Get star mask catalogue only if "masks" exist in both training and test datasets
    star_masks = (
    extract_star_data(data=data, train_key="masks", test_key="masks")
    if (
        data.training_data.dataset.get("masks") is not None 
        and data.test_data.dataset.get("masks") is not None
        and tf.size(data.training_data.dataset["masks"]) > 0  
        and tf.size(data.test_data.dataset["masks"]) > 0 
    )
    else None
    )

    pix_sampling = model_params.pix_sampling * 1e-6  # Change units from [um] to [m]

    # Ensure star_masks is properly handled
    star_masks = star_masks if star_masks is not None else [None] * len(star_postage_stamps)

    # Compute required Zernike 1 and Zernike 2
    zk1_2_array = -1.0 * compute_zernike_tip_tilt(
        star_postage_stamps, star_masks, pix_sampling, model_params.reference_shifts
    )

    # Zero pad array to get shape (n_stars, n_zernike=3)
    zernike_centroid_array = np.pad(
        zk1_2_array, pad_width=[(0, 0), (1, 0)], mode="constant", constant_values=0
    )

    return zernike_centroid_array


def compute_ccd_misalignment(model_params, data):
    """Compute CCD misalignment.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Object containing parameters for this PSF model class.
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    zernike_ccd_misalignment_array : np.ndarray
        Numpy array containing the Zernike contributions to model the CCD chip misalignments.
    """
    obs_positions = get_np_obs_positions(data)

    ccd_misalignment_calculator = CCDMisalignmentCalculator(
        tiles_path=model_params.ccd_misalignments_input_path,
        x_lims=model_params.x_lims,
        y_lims=model_params.y_lims,
        tel_focal_length=model_params.tel_focal_length,
        tel_diameter=model_params.tel_diameter,
    )
    # Compute required zernike 4 for each position
    zk4_values = np.array(
        [
            ccd_misalignment_calculator.get_zk4_from_position(single_pos)
            for single_pos in obs_positions
        ]
    ).reshape(-1, 1)

    # Zero pad array to get shape (n_stars, n_zernike=4)
    zernike_ccd_misalignment_array = np.pad(
        zk4_values, pad_width=[(0, 0), (3, 0)], mode="constant", constant_values=0
    )

    return zernike_ccd_misalignment_array


def get_zernike_prior(model_params, data):
    """Get Zernike priors from the provided dataset.

    This method concatenates the Zernike priors from both the training
    and test datasets.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Object containing parameters for this PSF model class.
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
    # Get hold of the simPSF parameters.
    # We need to add them to the config files

    # List of zernike contribution
    zernike_contribution_list = []

    if model_params.use_prior:
        logger.info("Reading in Zernike prior into Zernike contribution list...")
        zernike_contribution_list.append(get_np_zernike_prior(data))

    if model_params.correct_centroids:
        logger.info("Adding centroid correction to Zernike contribution list...")
        zernike_contribution_list.append(
            compute_centroid_correction(model_params, data)
        )

    if model_params.add_ccd_misalignments:
        logger.info("Adding CCD mis-alignments to Zernike contribution list...")
        zernike_contribution_list.append(compute_ccd_misalignment(model_params, data))

    if len(zernike_contribution_list) == 1:
        zernike_contribution = zernike_contribution_list[0]
    else:
        # Get max zk order
        max_zk_order = np.max(
            np.array(
                [
                    zk_contribution.shape[1]
                    for zk_contribution in zernike_contribution_list
                ]
            )
        )

        zernike_contribution = np.zeros(
            (zernike_contribution_list[0].shape[0], max_zk_order)
        )

        # Pad arrays to get the same length and add the final contribution
        for it in range(len(zernike_contribution_list)):
            current_zk_order = zernike_contribution_list[it].shape[1]
            current_zernike_contribution = np.pad(
                zernike_contribution_list[it],
                pad_width=[(0, 0), (0, int(max_zk_order - current_zk_order))],
                mode="constant",
                constant_values=0,
            )

            zernike_contribution += current_zernike_contribution

    return tf.convert_to_tensor(zernike_contribution, dtype=tf.float32)
