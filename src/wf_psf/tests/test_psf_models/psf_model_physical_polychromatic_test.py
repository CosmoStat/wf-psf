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
    zks_prior_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    return zks_prior_data


@pytest.fixture
def mock_data(mocker):
    mock_instance = mocker.Mock(spec=DataConfigHandler)
    # Configure the mock data object to have the necessary attributes
    mock_instance.training_data = mocker.Mock()
    mock_instance.training_data.dataset = {"positions": np.array([[1, 2], [3, 4]])}
    mock_instance.test_data = mocker.Mock()
    mock_instance.test_data.dataset = {"positions": np.array([[5, 6], [7, 8]])}
    return mock_instance

@pytest.fixture
def mock_model_params(mocker):
    model_params_mock = mocker.MagicMock()
    model_params_mock.param_hparams.n_zernikes = 10
    model_params_mock.zks_prior = [
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [7, 8, 9, 8],
        [10, 11, 12, 13],
    ]
    model_params_mock.pupil_diameter = 256
    return model_params_mock

def test_initialize_parameters(mocker, mock_data, mock_model_params):
    # Create mock objects for model_params, training_params
    #model_params_mock = mocker.MagicMock()
    mock_training_params = mocker.Mock()
    
    # Create TFPhysicalPolychromaticField instance
    field_instance = TFPhysicalPolychromaticField(
        mock_model_params, mock_training_params, mock_data
    )

    # Mock internal methods called during initialization
    mocker.patch(
        "wf_psf.data.training_preprocessing.get_obs_positions", return_value=True
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
        mock_model_params
    )
    field_instance._initialize_layers.assert_called_once_with(
        mock_model_params, mock_training_params
    )
    field_instance.assign_coeff_matrix.assert_not_called()  # Because coeff_mat is None in this test


def test_initialize_zernike_parameters(mocker, mock_model_params, mock_data):
    # Create training params mock object
    mock_training_params = mocker.Mock()

    # Create TFPhysicalPolychromaticField instance
    field_instance = TFPhysicalPolychromaticField(
        mock_model_params, mock_training_params, mock_data
    )

    # Assert that the attributes are set correctly
    assert field_instance.n_zernikes == mock_model_params.param_hparams.n_zernikes
    assert field_instance.zks_prior == mock_model_params.zks_prior
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
    field_instance._initialize_zernike_parameters(mock_model_params)
    
    assert field_instance.n_zks_total == tf.cast(
        tf.shape(mock_model_params.zks_prior)[1], tf.int32
    )
    # Expected shape of the tensor based on the input parameters
    expected_shape = (
        field_instance.n_zks_total,
        mock_model_params.pupil_diameter,
        mock_model_params.pupil_diameter,
    )
    assert field_instance.zernike_maps.shape == expected_shape
    
@pytest.mark.skip(reason="Need to fix bug.")
def test_initialize_physical_layer_mocking(mocker, mock_model_params, mock_data):
    # Create training params mock object
    mock_training_params = mocker.Mock()
    
    # Create a mock for the TFPhysicalLayer class
    mock_physical_layer_class = mocker.patch("wf_psf.psf_models.tf_layers.TFPhysicalLayer")
  
    # Create TFPhysicalPolychromaticField instance
    field_instance = TFPhysicalPolychromaticField(
        mock_model_params, mock_training_params, mock_data
    )

    # Call the _initialize_physical_layer method
    field_instance._initialize_physical_layer(mock_model_params)

    # Assert that the TFPhysicalLayer class was called with the expected arguments
    mock_physical_layer_class.assert_called_once_with(
        field_instance.obs_pos,
        field_instance.zks_prior,
        interpolation_type=mock_model_params.interpolation_type,
        interpolation_args=mock_model_params.interpolation_args,
    )

