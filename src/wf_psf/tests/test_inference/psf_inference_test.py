"""UNIT TESTS FOR PACKAGE MODULE: PSF Inference.

This module contains unit tests for the wf_psf.inference.psf_inference module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import os
from pathlib import Path
import pytest
import tensorflow as tf
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from wf_psf.inference.psf_inference import (
    InferenceConfigHandler, 
    PSFInference,
    PSFInferenceEngine 
)

from wf_psf.utils.read_config import RecursiveNamespace

@pytest.fixture
def mock_training_config():
    training_config = RecursiveNamespace(
        training=RecursiveNamespace(
            id_name="mock_id",
            model_params=RecursiveNamespace(
                model_name="mock_model",
                output_Q=2,
                output_dim=32,
                pupil_diameter=256,
                oversampling_rate=3,
                interpolation_type=None,
                interpolation_args=None,
                sed_interp_pts_per_bin=0,
                sed_extrapolate=True,
                sed_interp_kind="linear",
                sed_sigma=0,
                x_lims=[0.0, 1000.0],
                y_lims=[0.0, 1000.0],
                pix_sampling=12,
                tel_diameter=1.2,
                tel_focal_length=24.5,
                euclid_obsc=True,
                LP_filter_length=3,
                param_hparams=RecursiveNamespace(
                    n_zernikes=10,
                    
                )
            )  
        )   
    )
    return training_config

@pytest.fixture
def mock_inference_config():
    inference_config = RecursiveNamespace(
        inference=RecursiveNamespace(
            batch_size=16,
            cycle=2,
            configs=RecursiveNamespace(
                trained_model_path='/path/to/trained/model',
                model_subdir='psf_model',
                trained_model_config_path='config/training_config.yaml',
                data_config_path=None
            ),
            model_params=RecursiveNamespace(
                n_bins_lda=8,
                output_Q=1,
                output_dim=64
            ),
        )
    )
    return inference_config


@pytest.fixture
def psf_test_setup(mock_inference_config):
    num_sources = 2
    num_bins = 10
    output_dim = 32

    mock_positions = tf.convert_to_tensor([[0.1, 0.1], [0.2, 0.2]], dtype=tf.float32)
    mock_seds = tf.convert_to_tensor(np.random.rand(num_sources, 2, num_bins), dtype=tf.float32)
    expected_psfs = np.random.rand(num_sources, output_dim, output_dim).astype(np.float32)

    inference = PSFInference(
        "dummy_path.yaml",
        x_field=[0.1, 0.2],
        y_field=[0.1, 0.2],
        seds=np.random.rand(num_sources, num_bins)
    )
    inference._config_handler = MagicMock()
    inference._config_handler.inference_config = mock_inference_config
    inference._trained_psf_model = MagicMock()

    return {
        "inference": inference,
        "mock_positions": mock_positions,
        "mock_seds": mock_seds,
        "expected_psfs": expected_psfs,
        "num_sources": num_sources,
        "num_bins": num_bins,
        "output_dim": output_dim
    }


def test_set_config_paths(mock_inference_config):
    """Test setting configuration paths."""

    # Initialize handler and inject mock config
    config_handler = InferenceConfigHandler("fake/path")
    config_handler.inference_config = mock_inference_config

    # Call the method under test
    config_handler.set_config_paths()

    # Assertions
    assert config_handler.trained_model_path == Path("/path/to/trained/model")
    assert config_handler.model_subdir == "psf_model"
    assert config_handler.trained_model_config_path == Path("/path/to/trained/model/config/training_config.yaml")
    assert config_handler.data_config_path == None


def test_overwrite_model_params(mock_training_config, mock_inference_config):
    """Test that model_params can be overwritten."""
    # Mock the model_params object with some initial values
    training_config = mock_training_config
    inference_config = mock_inference_config

    InferenceConfigHandler.overwrite_model_params(
        training_config, inference_config
    )

    # Assert that the model_params were overwritten correctly
    assert training_config.training.model_params.output_Q == 1, "output_Q should be overwritten"
    assert training_config.training.model_params.output_dim == 64, "output_dim should be overwritten"   
  
    assert training_config.training.id_name == "mock_id", "id_name should not be overwritten"


def test_prepare_configs(mock_training_config, mock_inference_config):
    """Test preparing configurations for inference."""
    # Mock the model_params object with some initial values
    training_config = mock_training_config
    inference_config = mock_inference_config

    # Make copy of the original training config model_params
    original_model_params = mock_training_config.training.model_params

    # Instantiate PSFInference
    psf_inf = PSFInference('/dummy/path.yaml')

    # Mock the config handler attribute with a mock InferenceConfigHandler
    mock_config_handler = MagicMock(spec=InferenceConfigHandler)
    mock_config_handler.training_config = training_config
    mock_config_handler.inference_config = inference_config

    # Patch the overwrite_model_params to use the real static method
    mock_config_handler.overwrite_model_params.side_effect = InferenceConfigHandler.overwrite_model_params

    psf_inf._config_handler = mock_config_handler

    # Run prepare_configs
    psf_inf.prepare_configs()

    # Assert that the training model_params were updated
    assert original_model_params.output_Q == 1
    assert original_model_params.output_dim == 64


def test_config_handler_lazy_load(monkeypatch):
    inference = PSFInference("dummy_path.yaml")

    called = {}

    class DummyHandler:
        def load_configs(self):
            called['load'] = True
            self.inference_config = {}
            self.training_config = {}
            self.data_config = {}
        def overwrite_model_params(self, *args): pass

    monkeypatch.setattr("wf_psf.inference.psf_inference.InferenceConfigHandler", lambda path: DummyHandler())

    inference.prepare_configs()

    assert 'load' in called  # Confirm lazy load happened

def test_batch_size_positive():
    inference = PSFInference("dummy_path.yaml")
    inference._config_handler = MagicMock()
    inference._config_handler.inference_config = SimpleNamespace(
        inference=SimpleNamespace(batch_size=4, model_params=SimpleNamespace(output_dim=32))
    )
    assert inference.batch_size == 4


@patch('wf_psf.inference.psf_inference.DataHandler')
@patch('wf_psf.inference.psf_inference.load_trained_psf_model')
def test_load_inference_model(mock_load_trained_psf_model, mock_data_handler, mock_training_config, mock_inference_config):
    mock_data_config = MagicMock()
    mock_data_handler.return_value = mock_data_config
    mock_config_handler = MagicMock(spec=InferenceConfigHandler)
    mock_config_handler.trained_model_path = "mock/path/to/model"
    mock_config_handler.training_config = mock_training_config
    mock_config_handler.inference_config = mock_inference_config
    mock_config_handler.model_subdir = "psf_model"
    mock_config_handler.data_config = MagicMock()
  
    psf_inf = PSFInference("dummy_path.yaml")
    psf_inf._config_handler = mock_config_handler

    psf_inf.load_inference_model()

    weights_path_pattern = os.path.join(
            mock_config_handler.trained_model_path,
            mock_config_handler.model_subdir,
            f"{mock_config_handler.model_subdir}*_{mock_config_handler.training_config.training.model_params.model_name}*{mock_config_handler.training_config.training.id_name}_cycle{mock_config_handler.inference_config.inference.cycle}*"
        )

    # Assert calls to the mocked methods
    mock_load_trained_psf_model.assert_called_once_with(
        mock_training_config,
        mock_data_config,
        weights_path_pattern
    )

@patch.object(PSFInference, 'prepare_configs')
@patch.object(PSFInference, '_prepare_positions_and_seds')
@patch.object(PSFInferenceEngine, 'compute_psfs')
def test_run_inference(mock_compute_psfs, mock_prepare_positions_and_seds, mock_prepare_configs,  psf_test_setup):
    inference = psf_test_setup["inference"]
    mock_positions = psf_test_setup["mock_positions"]
    mock_seds = psf_test_setup["mock_seds"]
    expected_psfs = psf_test_setup["expected_psfs"]

    mock_prepare_positions_and_seds.return_value = (mock_positions, mock_seds)
    mock_compute_psfs.return_value = expected_psfs

    psfs = inference.run_inference()

    assert isinstance(psfs, np.ndarray)
    assert psfs.shape == expected_psfs.shape
    mock_prepare_positions_and_seds.assert_called_once()
    mock_compute_psfs.assert_called_once_with(mock_positions, mock_seds)
    mock_prepare_configs.assert_called_once()

@patch("wf_psf.inference.psf_inference.psf_models.simPSF")
def test_simpsf_uses_updated_model_params(mock_simpsf, mock_training_config, mock_inference_config):
    """Test that simPSF uses the updated model parameters."""
    training_config = mock_training_config
    inference_config = mock_inference_config

    # Set the expected output_Q
    expected_output_Q = inference_config.inference.model_params.output_Q
    training_config.training.model_params.output_Q = expected_output_Q

    # Create fake psf instance
    fake_psf_instance = MagicMock()
    fake_psf_instance.output_Q = expected_output_Q
    mock_simpsf.return_value = fake_psf_instance

    mock_config_handler = MagicMock(spec=InferenceConfigHandler)
    mock_config_handler.trained_model_path = "mock/path/to/model"
    mock_config_handler.training_config = training_config
    mock_config_handler.inference_config = inference_config
    mock_config_handler.model_subdir = "psf_model"
    mock_config_handler.data_config = MagicMock()
  
    modeller = PSFInference("dummy_path.yaml")
    modeller._config_handler = mock_config_handler

    modeller.prepare_configs()
    result = modeller.simPSF

    # Confirm simPSF was called once with the updated model_params
    mock_simpsf.assert_called_once()
    called_args, _ = mock_simpsf.call_args
    model_params_passed = called_args[0]
    assert model_params_passed.output_Q == expected_output_Q
    assert result.output_Q == expected_output_Q


@patch.object(PSFInference, '_prepare_positions_and_seds')
@patch.object(PSFInferenceEngine, 'compute_psfs')
def test_get_psfs_runs_inference(mock_compute_psfs, mock_prepare_positions_and_seds, psf_test_setup):
    inference = psf_test_setup["inference"]
    mock_positions = psf_test_setup["mock_positions"]
    mock_seds = psf_test_setup["mock_seds"]
    expected_psfs = psf_test_setup["expected_psfs"]

    mock_prepare_positions_and_seds.return_value = (mock_positions, mock_seds)

    def fake_compute_psfs(positions, seds):
        inference.engine._inferred_psfs = expected_psfs
        return expected_psfs

    mock_compute_psfs.side_effect = fake_compute_psfs

    psfs_1 = inference.get_psfs()
    assert np.all(psfs_1 == expected_psfs)

    psfs_2 = inference.get_psfs()
    assert np.all(psfs_2 == expected_psfs)

    assert mock_compute_psfs.call_count == 1
