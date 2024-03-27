"""UNIT TESTS FOR PACKAGE MODULE: PSF MODELS.

This module contains unit tests for the wf_psf.psf_models psf_models module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

import pytest
from wf_psf.psf_models import psf_models
from wf_psf.utils.io import FileIOHandler
import tensorflow as tf
import numpy as np
import os


def test_get_psf_model_weights_filepath():
    weights_filepath = "src/wf_psf/tests/data/validation/main_random_seed/checkpoint/checkpoint*_poly*_sample_w_bis1_2k_cycle2*"

    ans = psf_models.get_psf_model_weights_filepath(weights_filepath)
    assert (
        ans
        == "src/wf_psf/tests/data/validation/main_random_seed/checkpoint/checkpoint_callback_poly_sample_w_bis1_2k_cycle2"
    )


def test_generate_zernike_maps_3d():
    # Define test parameters
    n_zernikes = 5
    pupil_diam = 10

    # Call the function to generate Zernike maps
    zernike_maps = psf_models.generate_zernike_maps_3d(n_zernikes, pupil_diam)

    # Assertions to verify properties of the returned tensor
    assert isinstance(
        zernike_maps, tf.Tensor
    )  # Check if the returned value is a TensorFlow tensor
    assert (
        zernike_maps.dtype == tf.float32
    )  # Check if the data type of the tensor is float32

    # Expected shape of the tensor based on the input parameters
    expected_shape = (n_zernikes, pupil_diam, pupil_diam)
    assert (
        zernike_maps.shape == expected_shape
    )  # Check if the shape of the tensor matches the expected shape

    # Check if all values in the tensor are within the expected range (e.g., between 0 and 1)
    # assert tf.reduce_all(tf.logical_and(zernike_maps >= 0, zernike_maps <= 1))  #Fails
