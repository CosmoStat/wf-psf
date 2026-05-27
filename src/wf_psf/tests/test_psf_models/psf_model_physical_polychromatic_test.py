"""UNIT TESTS FOR PACKAGE MODULE: psf_model_physical_polychromatic.

This module contains unit tests for the wf_psf.psf_models.psf_model_physical_polychromatic module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
import numpy as np
import tensorflow as tf
from wf_psf.psf_models.models.psf_model_physical_polychromatic import (
    TFPhysicalPolychromaticField,
)


@pytest.fixture
def zks_prior():
    # Define your zks_prior data here
    zks_prior_data = [
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [7, 8, 9, 8],
        [10, 11, 12, 13],
    ]
    return tf.convert_to_tensor(zks_prior_data, dtype=tf.float32)


@pytest.fixture
def mock_data(zks_prior):
    return {
        "positions": np.array([[1, 2], [3, 4]]),
        "zernike_prior": zks_prior,
        "sources": np.zeros((2, 1, 1, 1)),
    }


@pytest.fixture
def mock_model_params(mocker):
    model_params_mock = mocker.MagicMock()

    model_params_mock.random_seed = 42
    model_params_mock.param_hparams.n_zernikes = 10
    model_params_mock.pupil_diameter = 256

    return model_params_mock


def test_compute_zernikes(mocker, mock_model_params, training_params, mock_data):
    # Generate patches
    mocker.patch(
        "wf_psf.psf_models.models.psf_model_physical_polychromatic.TFPhysicalPolychromaticField._assemble_zernike_contributions",
        return_value=tf.constant([[[[1.0]]], [[[2.0]]]]),
    )
    mocker.patch(
        "wf_psf.psf_models.models.psf_model_physical_polychromatic.TFPolynomialZernikeField",
        return_value=mocker.MagicMock(return_value=tf.constant([[[[1.0]]]])),
    )

    mocker.patch(
        "wf_psf.psf_models.models.psf_model_physical_polychromatic.TFNonParametricPolynomialVariationsOPD",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "wf_psf.psf_models.models.psf_model_physical_polychromatic.TFPhysicalLayer",
        return_value=mocker.MagicMock(return_value=tf.constant([[[[1.0]]]])),
    )

    # Expected output of mock components
    padded_zernike_params = tf.constant(
        [[[[10]], [[20]], [[30]], [[40]]]], dtype=tf.float32
    )
    padded_zernike_prior = tf.constant([[[[1]], [[2]], [[0]], [[0]]]], dtype=tf.float32)

    physical_layer_instance = TFPhysicalPolychromaticField(
        mock_model_params, training_params, mock_data
    )

    # tf_poly_Z_field was already instantiated as a mock; configure its call behaviour
    physical_layer_instance._tf_poly_Z_field = mocker.MagicMock(
        return_value=tf.constant([[[[1.0]]]])
    )
    # tf_physical_layer is used in compute_zernikes but built elsewhere — patch it
    mock_physical_layer = mocker.MagicMock()
    mock_physical_layer.call.return_value = tf.constant([[[[1.0]]]])
    physical_layer_instance._tf_physical_layer = mock_physical_layer

    # --- Expected values ---
    padded_zernike_params = tf.constant(
        [[[[10]], [[20]], [[30]], [[40]]]], dtype=tf.float32
    )
    padded_zernike_prior = tf.constant([[[[1]], [[2]], [[0]], [[0]]]], dtype=tf.float32)
    expected_values = tf.constant([[[[11]], [[22]], [[30]], [[40]]]], dtype=tf.float32)

    # Patch pad_tf_zernikes function
    mocker.patch(
        "wf_psf.psf_models.models.psf_model_physical_polychromatic.pad_tf_zernikes",
        return_value=(padded_zernike_params, padded_zernike_prior),
    )

    # Run the test
    zernike_coeffs = physical_layer_instance.compute_zernikes(tf.constant([[0.0, 0.0]]))

    # Assertions
    tf.debugging.assert_equal(zernike_coeffs, expected_values)
    assert zernike_coeffs.shape == expected_values.shape
