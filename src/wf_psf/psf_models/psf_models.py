"""PSF_Models.

A module which provides general utility methods
to manage the parameters of the psf model.

:Authors: Tobias Liaudat <tobiasliaudat@gmail.com> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from wf_psf.utils.utils import PI_zernikes, zernike_generator
from wf_psf.sims.psf_simulator import PSFSimulator
import glob
from sys import exit
import logging

logger = logging.getLogger(__name__)

PSF_FACTORY = {}


class PsfModelError(Exception):
    """PSF Model Parameter Error exception class for specific error scenarios."""

    def __init__(
        self, message="An error with your PSF model parameter settings occurred."
    ):
        self.message = message
        super().__init__(self.message)


class PSFModelBaseFactory:
    def get_model_instance(self,model_params, training_params, data=None, coeff_matrix=None):
        pass
    
def register_psfclass(psf_factory_class):
    """Register PSF Factory Class.

    A function to register a PSF factory class in a dictionary.

    Parameters
    ----------
    psf_identifier: str
        Identifier for the PSF model
    factory_class: type
        PSF Factory Class

    """
    for id in psf_factory_class.ids:
        PSF_FACTORY[id] = psf_factory_class


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
        raise PsfModelError("PSF model entered is invalid. Check your config settings.")
    return psf_factory_class


def get_psf_model(*psf_model_params):
    """Get PSF Model Class Instance.

    A function to instantiate a
    PSF model class.

    Parameters
    ----------
    model_id: str
        PSF model identifier
    model_params: type
        Recursive Namespace object
    training_hparams: type
        Recursive Namespace object
    coeff_matrix: Tensor or None, optional
        Initialization of the coefficient matrix defining the parametric psf field model

    Returns
    -------
    psf_class: class instance
        PSF model class instance

    """
  
    model_name = psf_model_params[0].model_name
    psf_class = set_psf_model(model_name)
    psf_factory_class = PSF_FACTORY.get(model_name)
    if psf_factory_class is None:
        raise PsfModelError("PSF model entered is invalid. Check your config settings.")

    return psf_factory_class().create_instance(*psf_model_params)


def get_psf_model_weights_filepath(weights_filepath):
    """Get PSF model weights filepath.

    A function to return the basename of the user-specified psf model weights path.

    Parameters
    ----------
    weights_filepath: str
        Basename of the psf model weights to be loaded.

    Returns
    -------
    str
        The absolute path concatenated to the basename of the psf model weights to be loaded.

    """
    try:
        return glob.glob(weights_filepath)[0].split(".")[0]
    except IndexError:
        logger.exception(
            "PSF weights file not found. Check that you've specified the correct weights file in the metrics config file."
        )
        raise PsfModelError("PSF model weights error.")


def tf_zernike_cube(n_zernikes, pupil_diam):
    """Tensor Flow Zernike Cube.

    A function to generate Zernike maps on
    a three-dimensional tensor.

    Parameters
    ----------
    n_zernikes: int
        Number of Zernike polynomials
    pupil_diam: float
        Size of the pupil diameter

    Returns
    -------
    Zernike map tensor
        TensorFlow EagerTensor type

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


def tf_obscurations(pupil_diam, N_filter=2):
    """Tensor Flow Obscurations.

    A function to generate obscurations as a tensor.

    Parameters
    ----------
    pupil_diam: float
        Size of the pupil diameter
    N_filters: int
        Number of filters

    Returns
    -------
    Obscurations tensor
        TensorFlow EagerTensor type

    """
    obscurations = PSFSimulator.generate_pupil_obscurations(
        N_pix=pupil_diam, N_filter=N_filter
    )
    return tf.convert_to_tensor(obscurations, dtype=tf.complex64)

    ## Generate initializations -- This looks like it could be moved to PSF model package
    # Prepare np input


def simPSF(model_params):
    """Simulated PSF model.

    A function to instantiate a
    simulated PSF model object.

    Features
    --------
    model_params: Recursive Namespace object
        Recursive Namespace object storing model parameters

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
    )

    simPSF_np.gen_random_Z_coeffs(max_order=model_params.param_hparams.n_zernikes)
    z_coeffs = simPSF_np.normalize_zernikes(
        simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms
    )
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)

    return simPSF_np
