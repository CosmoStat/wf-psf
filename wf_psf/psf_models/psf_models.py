"""PSF_Models.

A module which provides general utility methods
to manage the parameters of the psf model.

:Authors: Tobias Liaudat <tobiasliaudat@gmail.com> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from wf_psf.utils.utils import PI_zernikes, zernike_generator
from wf_psf.sims.SimPSFToolkit import SimPSFToolkit
import logging

logger = logging.getLogger(__name__)

PSF_CLASS = {}


class PsfModelError(Exception):
    pass


def register_psfclass(psf_class):
    """Register PSF Class.

    A wrapper function to register all PSF model classes
    in a dictionary.

    Parameters
    ----------
    psf_class: type
        PSF Class

    Returns
    -------
    psf_class: type
        PSF class

    """
    for id in psf_class.ids:
        PSF_CLASS[id] = psf_class

    return psf_class


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
        psf_class = PSF_CLASS[model_name]
    except KeyError as e:
        logger.exception("PSF model entered is invalid. Check your config settings.")
        exit()

    return psf_class


def get_psf_model(model_params, training_hparams, *coeff_matrix):
    """Get PSF Model Class Instance.

    A function to instantiate a
    PSF model class.

    Parameters
    ----------
    model_name: str
        Short name of PSF model
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
    psf_class = set_psf_model(model_params.model_name)

    return psf_class(model_params, training_hparams, *coeff_matrix)


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


def tf_obscurations(pupil_diam, model, N_filter=2):
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
    simPSF_np = simPSF(model)
    obscurations = simPSF_np.generate_pupil_obscurations(
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

    simPSF_np = SimPSFToolkit(
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
    z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)

    return simPSF_np
