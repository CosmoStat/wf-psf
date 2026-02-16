"""PSF_Models.

A module which provides general utility methods
to manage the parameters of the psf model.

:Authors: Tobias Liaudat <tobiasliaudat@gmail.com> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import tensorflow as tf
from wf_psf.sims.psf_simulator import PSFSimulator
from wf_psf.utils.utils import zernike_generator
from wf_psf.utils.optimizer import is_optimizer_instance, get_optimizer
import glob
import logging

logger = logging.getLogger(__name__)

PSF_FACTORY = {}


class PSFModelError(Exception):
    """PSF Model Parameter Error exception class.

    This exception class is used to handle errors related to PSF (Point Spread Function) model parameters.

    Parameters
    ----------
    message : str, optional
        Error message to be raised. Defaults to "An error with your PSF model parameter settings occurred."
    """

    def __init__(
        self, message="An error with your PSF model parameter settings occurred."
    ):
        self.message = message
        super().__init__(self.message)


def register_psfclass(psf_factory_class):
    """Register PSF Factory Class.

    A function to register a PSF factory class in a dictionary.

    Parameters
    ----------
    factory_class: type
        PSF Factory Class

    """
    for id in psf_factory_class.ids:
        PSF_FACTORY[id] = psf_factory_class
        logger.info(id, PSF_FACTORY)

    return psf_factory_class


class PSFModelBaseFactory:
    """Base factory class for PSF models.

    This class serves as the base factory for instantiating PSF (Point Spread Function) models.
    Subclasses should implement the `get_model_instance` method to provide specific PSF model instances.

    Attributes
    ----------
    None

    Methods
    -------
    get_model_instance(model_params, training_params, data=None, coeff_matrix=None)
        Instantiates a PSF model with the provided parameters.

    Notes
    -----
    Subclasses of `PSFModelBaseFactory` should override the `get_model_instance` method to provide
    implementation-specific logic for instantiating PSF model instances.
    """

    def get_model_instance(
        self, model_params, training_params, data=None, coeff_matrix=None
    ):
        """Instantiate a PSF model instance.

        Parameters
        ----------
        model_params: Recursive Namespace
            A Recursive Namespace object containing parameters for this PSF model class.
        training_params: Recursive Namespace
            A Recursive Namespace object containing training hyperparameters for this PSF model class.
        data: DataConfigHandler
            A DataConfigHandler object that provides access to training and test datasets, as well as prior knowledge like Zernike coefficients.
        coeff_mat: Tensor or None, optional
            Coefficient matrix defining the parametric PSF field model.

        Returns
        -------
        PSF model instance
            An instance of the PSF model.
        """
        pass


def set_psf_model(model_name):
    """Set PSF Model Class.

    A function to select a class of
    the PSF model from a dictionary.

    Parameters
    ----------
    model_name: str
        Name of PSF model

    Returns
    -------
    psf_class: class
        Name of PSF model class

    """
    try:
        psf_factory_class = PSF_FACTORY[model_name]
    except KeyError as e:
        logger.exception(e)
        raise PSFModelError("PSF model entered is invalid. Check your config settings.")
    return psf_factory_class


def get_psf_model(*psf_model_params):
    """Get PSF Model Class Instance.

    A function to instantiate a
    PSF model class.

    Parameters
    ----------
    *psf_model_params : tuple
        Positional arguments representing the parameters required to instantiate the PSF model.

    Returns
    -------
    PSF model class instance
        An instance of the PSF model class based on the provided parameters.


    """
    model_name = psf_model_params[0].model_name
    psf_factory_class = set_psf_model(model_name)
    if psf_factory_class is None:
        raise PSFModelError("PSF model entered is invalid. Check your config settings.")

    return psf_factory_class().get_model_instance(*psf_model_params)


def build_PSF_model(model_inst, optimizer=None, loss=None, metrics=None):
    """Define the model-compilation parameters.

    Specially the loss function, the optimizer and the metrics.
    """
    # Define model loss function
    if loss is None:
        loss = tf.keras.losses.MeanSquaredError()

    # Handle optimizer: either config object or a Keras optimizer instance
    if is_optimizer_instance(optimizer):
        pass
    else:
        optimizer = get_optimizer(optimizer_config=optimizer)
    
    # Define metric functions
    if metrics is None:
        metrics = [tf.keras.metrics.MeanSquaredError()]

    # Compile the model
    model_inst.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=False,
    )

    return model_inst


def get_psf_model_weights_filepath(weights_filepath):
    """Get PSF model weights filepath.

    A function to return the basename of the user-specified PSF model weights path.

    Parameters
    ----------
    weights_filepath: str
        Basename of the PSF model weights to be loaded.

    Returns
    -------
    str
        The absolute path concatenated to the basename of the PSF model weights to be loaded.

    """
    try:
        return glob.glob(weights_filepath)[0].split(".")[0]
    except IndexError:
        logger.exception(
            "PSF weights file not found. Check that you've specified the correct weights file in the your config file."
        )
        raise PSFModelError("PSF model weights error.")


def generate_zernike_maps_3d(n_zernikes, pupil_diam):
    """Generate 3D Zernike Maps.

    This function generates Zernike maps on a three-dimensional tensor.

    Parameters
    ----------
    n_zernikes : int
        The number of Zernike polynomials.
    pupil_diam : float
        The diameter of the pupil.

    Returns
    -------
    tf.Tensor
        A TensorFlow EagerTensor containing the Zernike map tensor.

    Notes
    -----
    The Zernike maps are generated using the specified number of Zernike
    polynomials and the size of the pupil diameter. The resulting tensor
    contains the Zernike maps in a three-dimensional format.
    """
    # Prepare the inputs
    # Generate Zernike maps
    zernikes = zernike_generator(n_zernikes=n_zernikes, wfe_dim=pupil_diam)
    # Now as cubes
    np_zernike_cube = np.zeros(
        (len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1])
    )

    for it in range(len(zernikes)):
        np_zernike_cube[it, :, :] = zernikes[it]

    np_zernike_cube[np.isnan(np_zernike_cube)] = 0

    return tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)


def tf_obscurations(pupil_diam, N_filter=2, rotation_angle=0):
    """Tensor Flow Obscurations.

    A function to generate obscurations as a tensor.

    Parameters
    ----------
    pupil_diam: float
        Size of the pupil diameter
    N_filters: int
        Number of filters
    rotation_angle: int
        Rotation angle in degrees to apply to the obscuration pattern. It only supports 90 degree rotations. The rotation will be counterclockwise.

    Returns
    -------
    Obscurations tensor
        TensorFlow EagerTensor type

    """
    obscurations = PSFSimulator.generate_euclid_pupil_obscurations(
        N_pix=pupil_diam, N_filter=N_filter, rotation_angle=rotation_angle
    )
    return tf.convert_to_tensor(obscurations, dtype=tf.complex64)


def simPSF(model_params):
    """Instantiate and configure a Simulated PSF model.

    This function creates a `PSFSimulator` instance with the given model parameters, generates random Zernike coefficients, normalizes them, and produces a monochromatic PSF.

    Parameters
    ----------
    model_params: Recursive Namespace
        A recursive namespace object storing model parameters

    Returns
    -------
    PSFSimulator
        A configured `PSFSimulator` instance with the specified model parameters.

    """
    simPSF_np = PSFSimulator(
        max_order=model_params.param_hparams.n_zernikes,
        pupil_diameter=model_params.pupil_diameter,
        output_dim=model_params.output_dim,
        oversampling_rate=model_params.oversampling_rate,
        output_Q=model_params.output_Q,
        SED_interp_pts_per_bin=model_params.sed_interp_pts_per_bin,
        SED_extrapolate=model_params.sed_extrapolate,
        SED_interp_kind=model_params.sed_interp_kind,
        SED_sigma=model_params.sed_sigma,
        pix_sampling=model_params.pix_sampling,
        tel_diameter=model_params.tel_diameter,
        tel_focal_length=model_params.tel_focal_length,
        euclid_obsc=model_params.euclid_obsc,
        LP_filter_length=model_params.LP_filter_length,
    )

    simPSF_np.gen_random_Z_coeffs(max_order=model_params.param_hparams.n_zernikes)
    z_coeffs = simPSF_np.normalize_zernikes(
        simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms
    )
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)

    return simPSF_np
