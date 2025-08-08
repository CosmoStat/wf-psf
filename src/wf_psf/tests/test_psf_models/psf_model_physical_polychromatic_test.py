"""UNIT TESTS FOR PACKAGE MODULE: psf_model_physical_polychromatic.

This module contains unit tests for the wf_psf.psf_models.psf_model_physical_polychromatic module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch
from wf_psf.psf_models.models.psf_model_physical_polychromatic import (
    TFPhysicalPolychromaticField,
)
from wf_psf.utils.configs_handler import DataConfigHandler


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
def mock_data(mocker, zks_prior):
    mock_instance = mocker.Mock(spec=DataConfigHandler)
    mock_instance.run_type = "training"
    
    training_dataset = {
        "positions": np.array([[1, 2], [3, 4]]),
        "zernike_prior": zks_prior,
        "noisy_stars": np.zeros((2, 1, 1, 1)), 
    }
    test_dataset = {
        "positions": np.array([[5, 6], [7, 8]]),
        "zernike_prior": zks_prior,
        "stars": np.zeros((2, 1, 1, 1)), 
    }

    mock_instance.training_data = mocker.Mock()
    mock_instance.training_data.dataset = training_dataset
    mock_instance.test_data = mocker.Mock()
    mock_instance.test_data.dataset = test_dataset
    mock_instance.batch_size = 16

    return mock_instance


@pytest.fixture
def mock_model_params(mocker):
    model_params_mock = mocker.MagicMock()
    model_params_mock.param_hparams.n_zernikes = 10
    model_params_mock.pupil_diameter = 256
    return model_params_mock

@pytest.fixture
def physical_layer_instance(mocker, mock_model_params, mock_data):
    # Patch expensive methods during construction to avoid errors
    with patch("wf_psf.psf_models.models.psf_model_physical_polychromatic.TFPhysicalPolychromaticField._assemble_zernike_contributions", return_value=tf.constant([[[[1.0]]], [[[2.0]]]])):
        from wf_psf.psf_models.models.psf_model_physical_polychromatic import TFPhysicalPolychromaticField
        instance = TFPhysicalPolychromaticField(mock_model_params, mocker.Mock(), mock_data)
        return instance

def test_compute_zernikes(mocker, physical_layer_instance):
    # Expected output of mock components
    padded_zernike_param = tf.constant([[[[10]], [[20]], [[30]], [[40]]]], dtype=tf.float32)
    padded_zernike_prior = tf.constant([[[[1]], [[2]], [[0]], [[0]]]], dtype=tf.float32)
    n_zks_total = physical_layer_instance.n_zks_total
    expected_values_list = [11, 22, 30, 40] + [0] * (n_zks_total - 4)
    expected_values = tf.constant(
        [[[[v]] for v in expected_values_list]],
        dtype=tf.float32
)
    # Patch tf_poly_Z_field method
    mocker.patch.object(
        TFPhysicalPolychromaticField,
        "tf_poly_Z_field",
        return_value=padded_zernike_param
    )

    # Patch tf_physical_layer.call method
    mock_tf_physical_layer = mocker.Mock()
    mock_tf_physical_layer.call.return_value = padded_zernike_prior
    mocker.patch.object(
        TFPhysicalPolychromaticField,
        "tf_physical_layer",
        mock_tf_physical_layer
    )

    # Patch pad_tf_zernikes function
    mocker.patch(
        "wf_psf.data.data_zernike_utils.pad_tf_zernikes",
        return_value=(padded_zernike_param, padded_zernike_prior)
    )

    # Run the test
    zernike_coeffs = physical_layer_instance.compute_zernikes(tf.constant([[0.0, 0.0]]))

    # Assertions
    tf.debugging.assert_equal(zernike_coeffs, expected_values)
    assert zernike_coeffs.shape == expected_values.shape
