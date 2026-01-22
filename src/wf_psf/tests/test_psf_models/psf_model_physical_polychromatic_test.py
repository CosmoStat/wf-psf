"""UNIT TESTS FOR PACKAGE MODULE: psf_model_physical_polychromatic.

This module contains unit tests for the wf_psf.psf_models.psf_model_physical_polychromatic module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import pytest
import numpy as np
import tensorflow as tf
from wf_psf.psf_models.psf_model_physical_polychromatic import (
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


def test_initialize_parameters(mocker, mock_data, mock_model_params, zks_prior):
    # Create mock objects for model_params, training_params
    # model_params_mock = mocker.MagicMock()
    mock_training_params = mocker.Mock()

    # Mock internal methods called during initialization
    mocker.patch(
        "wf_psf.psf_models.psf_model_physical_polychromatic.get_zernike_prior",
        return_value=zks_prior,
    )

    mocker.patch(
        "wf_psf.data.training_preprocessing.get_obs_positions", return_value=True
    )

    # Create TFPhysicalPolychromaticField instance
    field_instance = TFPhysicalPolychromaticField(
        mock_model_params, mock_training_params, mock_data
    )

    mocker.patch.object(field_instance, "_initialize_zernike_parameters")
    mocker.patch.object(field_instance, "_initialize_layers")
    mocker.patch.object(field_instance, "assign_coeff_matrix")

    # Call the method being tested
    field_instance._initialize_parameters_and_layers(
        mock_model_params, mock_training_params, mock_data
    )

    # Check if internal methods were called with the correct arguments
    field_instance._initialize_zernike_parameters.assert_called_once_with(
        mock_model_params, mock_data
    )
    field_instance._initialize_layers.assert_called_once_with(
        mock_model_params, mock_training_params
    )
    field_instance.assign_coeff_matrix.assert_not_called()  # Because coeff_mat is None in this test


def test_initialize_zernike_parameters(mocker, mock_model_params, mock_data, zks_prior):
    # Create training params mock object
    mock_training_params = mocker.Mock()

    # Mock internal methods called during initialization
    mocker.patch(
        "wf_psf.psf_models.psf_model_physical_polychromatic.get_zernike_prior",
        return_value=zks_prior,
    )

    # Create TFPhysicalPolychromaticField instance
    field_instance = TFPhysicalPolychromaticField(
        mock_model_params, mock_training_params, mock_data
    )

    # Assert that the attributes are set correctly
    # assert field_instance.n_zernikes == mock_model_params.param_hparams.n_zernikes
    assert np.array_equal(field_instance.zks_prior.numpy(), zks_prior.numpy())
    assert field_instance.n_zks_total == mock_model_params.param_hparams.n_zernikes
    assert isinstance(
        field_instance.zernike_maps, tf.Tensor
    )  # Check if the returned value is a TensorFlow tensor
    assert (
        field_instance.zernike_maps.dtype == tf.float32
    )  # Check if the data type of the tensor is float32

    # Expected shape of the tensor based on the input parameters
    expected_shape = (
        field_instance.n_zks_total,
        mock_model_params.pupil_diameter,
        mock_model_params.pupil_diameter,
    )
    assert field_instance.zernike_maps.shape == expected_shape

    # Modify model_params to simulate zks_prior > n_zernikes
    mock_model_params.param_hparams.n_zernikes = 2

    # Call the method again to initialize the parameters
    field_instance._initialize_zernike_parameters(mock_model_params, mock_data)

    assert field_instance.n_zks_total == tf.cast(
        tf.shape(field_instance.zks_prior)[1], tf.int32
    )
    # Expected shape of the tensor based on the input parameters
    expected_shape = (
        field_instance.n_zks_total,
        mock_model_params.pupil_diameter,
        mock_model_params.pupil_diameter,
    )
    assert field_instance.zernike_maps.shape == expected_shape


def test_initialize_physical_layer_mocking(
    mocker, mock_model_params, mock_data, zks_prior
):
    # Create training params mock object
    mock_training_params = mocker.Mock()

    # Mock internal methods called during initialization
    mocker.patch(
        "wf_psf.psf_models.psf_model_physical_polychromatic.get_zernike_prior",
        return_value=zks_prior,
    )

    # Create a mock for the TFPhysicalLayer class
    mock_physical_layer_class = mocker.patch(
        "wf_psf.psf_models.psf_model_physical_polychromatic.TFPhysicalLayer"
    )

    # Create TFPhysicalPolychromaticField instance
    field_instance = TFPhysicalPolychromaticField(
        mock_model_params, mock_training_params, mock_data
    )

    # Assert that the TFPhysicalLayer class was called with the expected arguments
    mock_physical_layer_class.assert_called_once_with(
        field_instance.obs_pos,
        field_instance.zks_prior,
        interpolation_type=mock_model_params.interpolation_type,
        interpolation_args=mock_model_params.interpolation_args,
    )


@pytest.fixture
def physical_layer_instance(mocker, mock_model_params, mock_data, zks_prior):
    # Create training params mock object
    mock_training_params = mocker.Mock()

    # Mock internal methods called during initialization
    mocker.patch(
        "wf_psf.psf_models.psf_model_physical_polychromatic.get_zernike_prior",
        return_value=zks_prior,
    )

    # Create a mock for the TFPhysicalLayer class
    mocker.patch("wf_psf.psf_models.psf_model_physical_polychromatic.TFPhysicalLayer")

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
    physical_layer_instance.n_zks_total = max(
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
    physical_layer_instance.n_zks_total = max(
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
    physical_layer_instance.n_zks_total = max(
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
    # Mock padded tensors
    padded_zk_param = tf.constant(
        [[[[10]], [[20]], [[30]], [[40]]]]
    )  # Shape: (1, 4, 1, 1)
    padded_zk_prior = tf.constant([[[[1]], [[2]], [[0]], [[0]]]])  # Shape: (1, 4, 1, 1)

    # Reset n_zks_total attribute
    physical_layer_instance.n_zks_total = 4  # Assuming a specific value for simplicity

    # Define the mock return values for tf_poly_Z_field and tf_physical_layer.call
    padded_zernike_param = tf.constant(
        [[[[10]], [[20]], [[30]], [[40]]]]
    )  # Shape: (1, 4, 1, 1)
    padded_zernike_prior = tf.constant(
        [[[[1]], [[2]], [[0]], [[0]]]]
    )  # Shape: (1, 4, 1, 1)

    mocker.patch.object(
        physical_layer_instance, "tf_poly_Z_field", return_value=padded_zk_param
    )
    mocker.patch.object(physical_layer_instance, "call", return_value=padded_zk_prior)
    mocker.patch.object(
        physical_layer_instance,
        "pad_zernikes",
        return_value=(padded_zernike_param, padded_zernike_prior),
    )

    # Call the method under test
    zernike_coeffs = physical_layer_instance.compute_zernikes(tf.constant([[0.0, 0.0]]))

    # Define the expected values
    expected_values = tf.constant(
        [[[[11]], [[22]], [[30]], [[40]]]]
    )  # Shape: (1, 4, 1, 1)

    # Assert that the shapes are equal
    assert zernike_coeffs.shape == expected_values.shape

    # Assert that the tensor values are equal
    assert tf.reduce_all(tf.equal(zernike_coeffs, expected_values))
