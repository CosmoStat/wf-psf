"""UNIT TESTS FOR PACKAGE MODULE: psf_model_physical_polychromatic.

This module contains unit tests for the wf_psf.psf_models.psf_model_physical_polychromatic module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import PropertyMock
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
def mock_data(mocker):
    mock_instance = mocker.Mock(spec=DataConfigHandler)
    # Configure the mock data object to have the necessary attributes
    mock_instance.run_type = "training"
    mock_instance.training_data = mocker.Mock()
    mock_instance.training_data.dataset = {"positions": np.array([[1, 2], [3, 4]])}
    mock_instance.test_data = mocker.Mock()
    mock_instance.test_data.dataset = {"positions": np.array([[5, 6], [7, 8]])}
    mock_instance.batch_size = 32
    return mock_instance


@pytest.fixture
def mock_model_params(mocker):
    model_params_mock = mocker.MagicMock()
    model_params_mock.param_hparams.n_zernikes = 10
    model_params_mock.pupil_diameter = 256
    return model_params_mock

@pytest.fixture
def physical_layer_instance(mocker, mock_model_params, mock_data, zks_prior):
    # Create training params mock object
    mock_training_params = mocker.Mock()

    # Create TFPhysicalPolychromaticField instance
    psf_field_instance = TFPhysicalPolychromaticField(
        mock_model_params, mock_training_params, mock_data
    )
    return psf_field_instance


def test_pad_zernikes_num_of_zernikes_equal(physical_layer_instance):
    # Define input tensors with same length and num of Zernikes
    zk_param = tf.constant([[[[1]]], [[[2]]]])  # Shape: (2, 1, 1, 1)
    zk_prior = tf.constant([[[[1]]], [[[2]]]])  # Shape: (2, 1, 1, 1)

    # Reshape the tensors to have the desired shapes
    zk_param = tf.reshape(zk_param, (1, 2, 1, 1))  # Reshaping tensor1 to (1, 2, 1, 1)
    zk_prior = tf.reshape(zk_prior, (1, 2, 1, 1))  # Reshaping tensor2 to (1, 2, 1, 1)

    # Reset n_zks_total attribute
    physical_layer_instance._n_zks_total = max(
         tf.shape(zk_param)[1].numpy(), tf.shape(zk_prior)[1].numpy()
    )
    # Call the method under test
    padded_zk_param, padded_zk_prior = physical_layer_instance.pad_zernikes(
        zk_param, zk_prior
    )

    # Assert that the padded tensors have the correct shapes
    assert padded_zk_param.shape == (1, 2, 1, 1)
    assert padded_zk_prior.shape == (1, 2, 1, 1)


def test_pad_zernikes_prior_greater_than_param(physical_layer_instance):
    zk_param = tf.constant([[[[1]]], [[[2]]]])  # Shape: (2, 1, 1, 1)
    zk_prior = tf.constant([[[[1]], [[2]], [[3]], [[4]], [[5]]]])  # Shape: (5, 1, 1, 1)

    # Reshape the tensors to have the desired shapes
    zk_param = tf.reshape(zk_param, (1, 2, 1, 1))  # Reshaping tensor1 to (1, 2, 1, 1)
    zk_prior = tf.reshape(zk_prior, (1, 5, 1, 1))  # Reshaping tensor2 to (1, 5, 1, 1)

    # Reset n_zks_total attribute
    physical_layer_instance._n_zks_total = max(
        tf.shape(zk_param)[1].numpy(), tf.shape(zk_prior)[1].numpy()
    )

    # Call the method under test
    padded_zk_param, padded_zk_prior = physical_layer_instance.pad_zernikes(
        zk_param, zk_prior
    )

    # Assert that the padded tensors have the correct shapes
    assert padded_zk_param.shape == (1, 5, 1, 1)
    assert padded_zk_prior.shape == (1, 5, 1, 1)


def test_pad_zernikes_param_greater_than_prior(physical_layer_instance):
    zk_param = tf.constant([[[[10]], [[20]], [[30]], [[40]]]])  # Shape: (4, 1, 1, 1)
    zk_prior = tf.constant([[[[1]]], [[[2]]]])  # Shape: (2, 1, 1, 1)

    # Reshape the tensors to have the desired shapes
    zk_param = tf.reshape(zk_param, (1, 4, 1, 1))  # Reshaping tensor1 to (1, 2, 1, 1)
    zk_prior = tf.reshape(zk_prior, (1, 2, 1, 1))  # Reshaping tensor2 to (1, 4, 1, 1)

    # Reset n_zks_total attribute
    physical_layer_instance._n_zks_total = max(
        tf.shape(zk_param)[1].numpy(), tf.shape(zk_prior)[1].numpy()
    )

    # Call the method under test
    padded_zk_param, padded_zk_prior = physical_layer_instance.pad_zernikes(
        zk_param, zk_prior
    )

    # Assert that the padded tensors have the correct shapes
    assert padded_zk_param.shape == (1, 4, 1, 1)
    assert padded_zk_prior.shape == (1, 4, 1, 1)


def test_compute_zernikes(mocker, physical_layer_instance):
  # Expected output of mock components
    padded_zernike_param = tf.constant([[[[10]], [[20]], [[30]], [[40]]]], dtype=tf.float32)
    padded_zernike_prior = tf.constant([[[[1]], [[2]], [[0]], [[0]]]], dtype=tf.float32)
    expected_values = tf.constant([[[[11]], [[22]], [[30]], [[40]]]], dtype=tf.float32)

    # Patch tf_poly_Z_field property
    mock_tf_poly_Z_field = mocker.Mock(return_value=padded_zernike_param)
    mocker.patch.object(
        TFPhysicalPolychromaticField,
        'tf_poly_Z_field',
        new_callable=PropertyMock,
        return_value=mock_tf_poly_Z_field
    )

    # Patch tf_physical_layer property
    mock_tf_physical_layer = mocker.Mock()
    mock_tf_physical_layer.call.return_value = padded_zernike_prior
    mocker.patch.object(
        TFPhysicalPolychromaticField,
        'tf_physical_layer',
        new_callable=PropertyMock,
        return_value=mock_tf_physical_layer
    )

    # Patch pad_zernikes instance method directly (this one isn't a property)
    mocker.patch.object(
        physical_layer_instance,
        'pad_zernikes',
        return_value=(padded_zernike_param, padded_zernike_prior)
    )


    # Run the test
    zernike_coeffs = physical_layer_instance.compute_zernikes(tf.constant([[0.0, 0.0]]))

    # Assertions
    tf.debugging.assert_equal(zernike_coeffs, expected_values)
    assert zernike_coeffs.shape == expected_values.shape