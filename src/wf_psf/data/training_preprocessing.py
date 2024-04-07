"""Training Data Processing.

A module to load and preprocess training and validation test data.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr> and Tobias Liaudat <tobiasliaudat@gmail.com>

"""

import os
import numpy as np
import wf_psf.utils.utils as utils
import tensorflow as tf
from wf_psf.utils.ccd_missalignments import CCDMissalignmentCalculator
from wf_psf.utils.centroids import get_zk1_2_for_observed_psf


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


def get_np_stars(data):
    """Get the full star catalogue from the provided dataset.

    This method concatenates the stars from both the training
    and test datasets to obtain the star catalogue.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    star_catalogue : np.ndarray
        Numpy array containing the full star catalogue.
    """
    
    star_catalogue = np.concatenate(
        (
            data.training_data.dataset['noisy_stars'].numpy(),
            data.test_data.dataset['stars'].numpy(),
        ),
        axis=0,
    )

    return star_catalogue

def get_np_zk_prior(data):
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

def compute_centroid_correction(data):
    """Compute centroid corrections.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    zernike_centroid_array : np.ndarray
        Numpy array containing the Zernike contributions to match WaveDiff model 
        centroid and the observed stars centroid.
    """

    star_catalogue = get_np_stars(data)

    pix_sampling = data.simPSF.pix_sampling * 1e-6  # Change units from [um] to [m]

    # Compute required Zernike 1 and Zernike 2
    # The -1 is to contrarest the actual shift
    zk1_2_array = -1. * np.array([
        get_zk1_2_for_observed_psf(obs_psf, pixel_sampling=pix_sampling)
        for obs_psf in star_catalogue
    ])

    # Zero pad array to get shape (n_stars, n_zernike=3)
    zernike_centroid_array = np.pad(
        zk1_2_array,
        pad_width=[(0,0), (1,0)],
        mode='constant',
        constant_values=0
    )

    return zernike_centroid_array
    


def compute_ccd_missalignment(model_params, data):
    """Compute CCD missalignment.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Object containing parameters for this PSF model class.
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    zernike_ccd_missalignment_array : np.ndarray
        Numpy array containing the Zernike contributions to model the CCD chip missalignments.
    """
    obs_positions = get_np_obs_positions(data)

    ccd_missalignment_calculator = CCDMissalignmentCalculator(
        tiles_path=model_params.ccd_missalignments_input_path,
        # tiles_path="/Users/tl255879/Documents/research/Euclid/real_data/CCD_missalignments/tiles.npy",
        x_lims=model_params.x_lims,
        y_lims=model_params.y_lims,
        tel_focal_length=data.simPSF.tel_focal_length,
        tel_diameter=data.simPSF.tel_diameter,
    )
    # Compute required zernike 4 for each position
    zk4_values = np.array([
        ccd_missalignment_calculator.get_zk4_from_position(single_pos)
        for single_pos in obs_positions
    ]).reshape(-1, 1)

    # Zero pad array to get shape (n_stars, n_zernike=4)
    zernike_ccd_missalignment_array = np.pad(
        zk4_values,
        pad_width=[(0,0), (3,0)],
        mode='constant',
        constant_values=0
    )

    return zernike_ccd_missalignment_array



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
        zernike_contribution_list.append(get_np_zk_prior(data))

    if model_params.correct_centroids:
        zernike_contribution_list.append(compute_centroid_correction(data))

    if model_params.add_ccd_missalignments:
        zernike_contribution_list.append(compute_ccd_missalignment(model_params, data))

    if len(zernike_contribution_list) == 1:
        zernike_contribution = zernike_contribution_list[0]
    else:
        # Get max zk order
        max_zk_order = np.max(np.array([
            zk_contribution.shape[1] for zk_contribution in zernike_contribution_list
        ]))

        zernike_contribution = np.zeros((zernike_contribution_list[0].shape[0], max_zk_order))

        # Pad arrays to get the same length and add the final contribution
        for it in range(len(zernike_contribution_list)):
            current_zk_order = zernike_contribution_list[it].shape[1]
            current_zernike_contribution = np.pad(
                zernike_contribution_list[it],
                pad_width=[(0, 0), (0, int(max_zk_order - current_zk_order))],
                mode='constant',
                constant_values=0,
            )

            zernike_contribution += current_zernike_contribution
            

    return tf.convert_to_tensor(zernike_contribution, dtype=tf.float32)
