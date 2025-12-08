"""PSF Model Loader.

This module provides helper functions for loading trained PSF models.
It includes utilities to:
- Load a model from disk using its configuration and weights.
- Prepare inputs for inference or evaluation workflows.

Author: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import logging
from wf_psf.psf_models.psf_models import get_psf_model, get_psf_model_weights_filepath

logger = logging.getLogger(__name__)


def load_trained_psf_model(training_conf, data_conf, weights_path_pattern):
    """
    Loads a trained PSF model and applies saved weights.

    Parameters
    ----------
    training_conf : RecursiveNamespace
        Configuration object containing model parameters and training hyperparameters.
        Supports attribute-style access to nested fields.
    data_conf : RecursiveNamespace or dict
        Configuration RecursiveNamespace object or a dictionary containing data parameters (e.g. pixel data, positions, masks, etc).
    weights_path_pattern : str
        Glob-style pattern used to locate the model weights file.

    Returns
    -------
    model : tf.keras.Model or compatible
        The PSF model instance with loaded weights.

    Raises
    ------
    RuntimeError
        If loading the model weights fails for any reason.
    """
    model = get_psf_model(
        training_conf.training.model_params,
        training_conf.training.training_hparams,
        data_conf,
    )

    weights_path = get_psf_model_weights_filepath(weights_path_pattern)

    try:
        logger.info(f"Loading PSF model weights from {weights_path}")
        status = model.load_weights(weights_path)
        status.expect_partial()

    except Exception as e:
        logger.exception("Failed to load model weights.")
        raise RuntimeError("Model weight loading failed.") from e
    return model
