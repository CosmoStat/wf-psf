"""UNIT TESTS FOR PACKAGE MODULE: PSF Inference.

This module contains unit tests for the wf_psf.inference.psf_inference module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import pytest
import tensorflow as tf
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from wf_psf.data.data_adapter import RepresentationState
from wf_psf.inference.psf_inference import (
    InferenceConfigHandler,
    PSFInference,
    PSFInferenceEngine,
)
from wf_psf.utils.read_config import RecursiveNamespace


#!!!!!! CONFIG FIXTURES !!!!!!!!!
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
                ),
            ),
        )
    )
    return training_config


@pytest.fixture
def mock_inference_config():
    inference_config = RecursiveNamespace(
        inference=RecursiveNamespace(
            batch_size=16,
            cycle=2,
            schema_mode="INFERENCE",
            configs=RecursiveNamespace(
                trained_model_path="/path/to/trained/model",
                model_subdir="psf_model",
                trained_model_config_path="config/training_config.yaml",
                data_config_path=None,
            ),
            model_params=RecursiveNamespace(
                n_bins_lambda=8,
                output_Q=1,
                output_dim=64,
                correct_centroids=False,
                add_ccd_misalignments=True,
            ),
        )
    )
    return inference_config


#!!!!!! DATASET FIXTURES !!!!!!!!!
@pytest.fixture
def mock_dataset():
    """
    Unified mock dataset fixture for positions, SEDs, and expected PSFs.
    Can be sliced or reshaped for single/multi-source tests.
    """
    dataset = {}

    # Multi-source example
    dataset["num_sources_multi"] = 2
    dataset["num_bins"] = 10
    dataset["output_dim"] = 32

    positions_multi = np.array([[0.1, 0.1], [0.2, 0.2]], dtype=np.float32)
    seds_multi = np.random.rand(
        dataset["num_sources_multi"], dataset["num_bins"], 2
    ).astype(np.float32)
    expected_psfs_multi = np.random.rand(
        dataset["num_sources_multi"], dataset["output_dim"], dataset["output_dim"]
    ).astype(np.float32)

    # Single-source example (reshaped or sliced)
    dataset["num_sources_single"] = 1
    positions_single = positions_multi[:1]
    seds_single = seds_multi[:1]
    expected_psfs_single = expected_psfs_multi[:1]

    # Convert to tensors
    dataset["positions_multi_tf"] = tf.convert_to_tensor(positions_multi)
    dataset["seds_multi_tf"] = tf.convert_to_tensor(seds_multi)
    dataset["expected_psfs_multi"] = expected_psfs_multi

    dataset["positions_single_tf"] = tf.convert_to_tensor(positions_single)
    dataset["seds_single_tf"] = tf.convert_to_tensor(seds_single)
    dataset["expected_psfs_single"] = expected_psfs_single

    return dataset


#!!!!!! ADAPTER FIXTURE !!!!!!!!!
@pytest.fixture
def mock_data_adapter(mock_dataset):
    """
    Adapter fixture that uses the unified mock dataset.
    """
    adapter = MagicMock()
    adapter.representation_state = RepresentationState.NUMPY
    adapter.convert_to_tensorflow = MagicMock()
    adapter.complete_data = {
        "positions": mock_dataset["positions_multi_tf"].numpy(),
        "seds": mock_dataset["seds_multi_tf"].numpy(),
    }
    return adapter


#!!!!!! PSF FIXTURES !!!!!!!!!
@pytest.fixture
def psf_test_setup(mock_dataset, mock_inference_config):
    inference = PSFInference(
        "dummy_path.yaml",
        x_field=[0.1, 0.2],
        y_field=[0.1, 0.2],
        seds=mock_dataset["seds_multi_tf"].numpy(),
    )
    inference._config_handler = MagicMock()
    inference._config_handler.inference_config = mock_inference_config.inference
    inference._trained_psf_model = MagicMock()

    return {
        "inference": inference,
        "mock_positions": mock_dataset["positions_multi_tf"],
        "mock_seds": mock_dataset["seds_multi_tf"],
        "expected_psfs": mock_dataset["expected_psfs_multi"],
        "num_sources": mock_dataset["num_sources_multi"],
        "num_bins": mock_dataset["num_bins"],
        "output_dim": mock_dataset["output_dim"],
    }


@pytest.fixture
def psf_single_star_setup(mock_dataset, mock_inference_config):
    inference = PSFInference(
        "dummy_path.yaml",
        x_field=0.1,
        y_field=0.1,
        seds=mock_dataset["seds_single_tf"].numpy()[0],  # shape (num_bins, 2)
    )
    inference._config_handler = MagicMock()
    inference._config_handler.inference_config = mock_inference_config.inference
    inference._trained_psf_model = MagicMock()

    return {
        "inference": inference,
        "mock_positions": mock_dataset["positions_single_tf"],
        "mock_seds": mock_dataset["seds_single_tf"],
        "expected_psfs": mock_dataset["expected_psfs_single"],
        "num_sources": mock_dataset["num_sources_single"],
        "num_bins": mock_dataset["num_bins"],
        "output_dim": mock_dataset["output_dim"],
    }


@pytest.fixture(params=["single", "multi"])
def psf_setup(mock_dataset, mock_inference_config, request):
    """
    Unified PSF setup fixture for both single-star and multi-star tests.
    Uses the unified mock_dataset.

    Parameters
    ----------
    request.param : str
        "single" for single-source setup, "multi" for multi-source setup.
    """
    if request.param == "multi":
        num_sources = mock_dataset["num_sources_multi"]
        positions_tf = mock_dataset["positions_multi_tf"]
        seds_tf = mock_dataset["seds_multi_tf"]
        expected_psfs = mock_dataset["expected_psfs_multi"]
        x_field = [float(x) for x in positions_tf[:, 0].numpy()]
        y_field = [float(y) for y in positions_tf[:, 1].numpy()]
        seds_input = seds_tf.numpy()
    else:
        num_sources = mock_dataset["num_sources_single"]
        positions_tf = mock_dataset["positions_single_tf"]
        seds_tf = mock_dataset["seds_single_tf"]
        expected_psfs = mock_dataset["expected_psfs_single"]
        x_field = float(positions_tf[0, 0].numpy())
        y_field = float(positions_tf[0, 1].numpy())
        seds_input = seds_tf.numpy()[0]  # shape (num_bins, 2)

    inference = PSFInference(
        "dummy_path.yaml",
        x_field=x_field,
        y_field=y_field,
        seds=seds_input,
    )
    inference._config_handler = MagicMock()
    inference._config_handler.inference_config = mock_inference_config.inference
    inference._trained_psf_model = MagicMock()

    return {
        "inference": inference,
        "mock_positions": positions_tf,
        "mock_seds": seds_tf,
        "expected_psfs": expected_psfs,
        "num_sources": num_sources,
        "num_bins": mock_dataset["num_bins"],
        "output_dim": mock_dataset["output_dim"],
        "mode": request.param,  # "single" or "multi"
    }


@pytest.fixture
def mock_compute_psfs_with_cache(psf_setup):
    """
    Fixture that patches PSFInferenceEngine.compute_psfs with a side effect
    that populates the engine's cache.
    Works for both single-star and multi-star setups.
    """
    inference = psf_setup["inference"]
    mock_positions = psf_setup["mock_positions"]
    mock_seds = psf_setup["mock_seds"]
    expected_psfs = psf_setup["expected_psfs"]

    with patch.object(PSFInferenceEngine, "compute_psfs") as mock_compute_psfs:

        def fake_compute_psfs(positions, seds):
            # Populate the engine cache with the expected PSFs
            inference.engine._inferred_psfs = expected_psfs
            return expected_psfs

        mock_compute_psfs.side_effect = fake_compute_psfs

        yield {
            "mock": mock_compute_psfs,
            "inference": inference,
            "positions": mock_positions,
            "seds": mock_seds,
            "expected_psfs": expected_psfs,
            "mode": psf_setup["mode"],  # "single" or "multi"
        }


def test_prepare_configs(mock_training_config, mock_inference_config):
    """Test preparing configurations for inference."""
    # Mock the model_params object with some initial values
    training_config = mock_training_config
    inference_config = mock_inference_config

    # Make copy of the original training config model_params
    original_model_params = mock_training_config.training.model_params

    # Instantiate PSFInference
    psf_inf = PSFInference("/dummy/path.yaml")

    # Mock the config handler attribute with a mock InferenceConfigHandler
    mock_config_handler = MagicMock(spec=InferenceConfigHandler)
    mock_config_handler.training_config = training_config.training
    mock_config_handler.inference_config = inference_config.inference

    # Patch the overwrite_model_params to use the real static method
    mock_config_handler.overwrite_model_params.side_effect = (
        InferenceConfigHandler.overwrite_model_params
    )

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
            called["load"] = True
            self.inference_config = {}
            self.training_config = {}
            self.data_config = {}

        def overwrite_model_params(self, *args):
            pass

    monkeypatch.setattr(
        "wf_psf.inference.psf_inference.InferenceConfigHandler",
        lambda path: DummyHandler(),
    )

    inference.prepare_configs()

    assert "load" in called  # Confirm lazy load happened


def test_batch_size_positive():
    inference = PSFInference("dummy_path.yaml")
    inference._config_handler = MagicMock()
    inference._config_handler.inference_config = SimpleNamespace(
        batch_size=4, model_params=SimpleNamespace(output_dim=32)
    )
    assert inference.batch_size == 4


def test_prepare_dataset_for_inference(psf_test_setup):
    inference = psf_test_setup["inference"]

    dataset = inference._prepare_dataset_for_inference()

    assert "positions" in dataset
    assert "seds" in dataset
    assert dataset["positions"].shape == (2, 2)


@patch("wf_psf.inference.psf_inference.DataAdapterFactory.build")
@patch("wf_psf.inference.psf_inference.psf_models.simPSF")
def test_data_adapter_property_adapter_build(
    _, mock_build, psf_test_setup, mock_data_adapter
):
    inference = psf_test_setup["inference"]

    mock_build.return_value = mock_data_adapter

    # Prevent real PSF simulator creation
    adapter = inference.inference_data_adapter

    assert adapter == mock_data_adapter
    mock_build.assert_called_once()


@patch("wf_psf.inference.psf_inference.DataAdapterFactory.build")
@patch("wf_psf.inference.psf_inference.psf_models.simPSF")
def test_data_adapter_cached(_, mock_build, psf_test_setup, mock_data_adapter):
    inference = psf_test_setup["inference"]
    mock_build.return_value = mock_data_adapter

    adapter1 = inference.inference_data_adapter
    adapter2 = inference.inference_data_adapter

    assert adapter1 is adapter2
    mock_build.assert_called_once()


@patch("wf_psf.inference.psf_inference.DataAdapterFactory.build")
def test_data_adapter_no_conversion_if_tensorflow(mock_build, psf_test_setup):
    adapter = MagicMock()
    adapter.representation_state = RepresentationState.TENSORFLOW
    adapter.convert_to_tensorflow = MagicMock()

    mock_build.return_value = adapter

    inference = psf_test_setup["inference"]

    inference.inference_data_adapter

    adapter.convert_to_tensorflow.assert_not_called()


@patch("wf_psf.inference.psf_inference.load_trained_psf_model")
def test_load_inference_model(
    mock_load_trained_psf_model,
    mock_training_config,
    mock_inference_config,
):
    mock_adapter = MagicMock()
    mock_adapter.complete_data = {"positions": np.zeros((2, 2))}

    psf_inf = PSFInference("dummy_path.yaml", x_field=2, y_field=2)

    mock_config_handler = MagicMock(spec=InferenceConfigHandler)
    mock_config_handler.trained_model_path = "mock/path/to/model"
    mock_config_handler.training_config = mock_training_config.training
    mock_config_handler.inference_config = mock_inference_config.inference
    mock_config_handler.model_subdir = "psf_model"

    psf_inf._config_handler = mock_config_handler
    psf_inf._model_data_adapter = mock_adapter

    psf_inf.load_inference_model()

    mock_load_trained_psf_model.assert_called_once()


def test_prepare_dataset_missing_positions():
    inference = PSFInference("dummy_path.yaml")

    with pytest.raises(ValueError):
        inference._prepare_dataset_for_inference()


@patch.object(PSFInference, "prepare_configs")
@patch.object(PSFInferenceEngine, "compute_psfs")
@patch("wf_psf.inference.psf_inference.DataAdapterFactory.build")
@patch("wf_psf.inference.psf_inference.psf_models.simPSF")
def test_run_inference(
    _,
    mock_build,
    mock_compute_psfs,
    mock_prepare_configs,
    mock_data_adapter,
    psf_test_setup,
):
    # Mock factory build
    mock_build.return_value = mock_data_adapter
    inference = psf_test_setup["inference"]

    # Lazy-load inference.data_adapter
    inference.inference_data_adapter

    mock_positions = mock_data_adapter.complete_data["positions"]
    mock_seds = mock_data_adapter.complete_data["seds"]
    expected_psfs = psf_test_setup["expected_psfs"]

    mock_compute_psfs.return_value = expected_psfs

    psfs = inference.run_inference()

    assert isinstance(psfs, np.ndarray)
    assert psfs.shape == expected_psfs.shape
    mock_prepare_configs.assert_called_once()
    mock_compute_psfs.assert_called_once_with(mock_positions, mock_seds)
    mock_data_adapter.convert_to_tensorflow.assert_called_once_with(
        inference.simPSF,
        inference.n_bins_lambda,
        mode=inference.config_handler.schema_mode,
    )


@patch("wf_psf.inference.psf_inference.psf_models.simPSF")
def test_simpsf_uses_updated_model_params(
    mock_simpsf, mock_training_config, mock_inference_config
):
    """Test that simPSF uses the updated model parameters."""
    training_config = mock_training_config.training
    inference_config = mock_inference_config.inference

    # Set the expected output_Q
    expected_output_Q = inference_config.model_params.output_Q
    training_config.model_params.output_Q = expected_output_Q

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


def test_get_psfs_runs_inference(mock_compute_psfs_with_cache):
    """Test that get_psfs uses cached PSFs after first computation."""
    inference = mock_compute_psfs_with_cache["inference"]
    expected_psfs = mock_compute_psfs_with_cache["expected_psfs"]

    inference.engine = MagicMock()
    inference.engine.inferred_psfs = None
    inference.engine.get_psfs.return_value = expected_psfs

    with patch.object(inference, "run_inference") as mock_run:

        def fake_run():
            inference.engine.inferred_psfs = expected_psfs
            return expected_psfs

        mock_run.side_effect = fake_run

        psfs_1 = inference.get_psfs()
        psfs_2 = inference.get_psfs()

        np.testing.assert_array_equal(psfs_1, expected_psfs)
        np.testing.assert_array_equal(psfs_2, expected_psfs)

        mock_run.assert_called_once()


def test_psf_shapes(psf_setup):
    setup = psf_setup
    psfs = setup["expected_psfs"]
    assert psfs.shape == (
        setup["num_sources"],
        setup["output_dim"],
        setup["output_dim"],
    )
    if setup["mode"] == "single":
        assert setup["num_sources"] == 1


def test_inference_clear_cache(psf_test_setup):
    """Test that PSFInference clear_cache resets the instance of PSFInference."""
    inference = psf_test_setup["inference"]
    inference._simPSF = MagicMock()
    inference._data_adapter = MagicMock()
    inference._trained_psf_model = MagicMock()
    inference._n_bins_lambda = MagicMock()
    inference._batch_size = MagicMock()
    inference._cycle = MagicMock()
    inference._output_dim = MagicMock()
    inference.engine = MagicMock()

    # Clear the cache
    inference.clear_cache()

    # Check that the internal cache is None
    assert inference._config_handler is None
    assert inference._simPSF is None
    assert inference._data_adapter is None
    assert inference._trained_psf_model is None
    assert inference._n_bins_lambda is None
    assert inference._batch_size is None
    assert inference._cycle is None
    assert inference._output_dim is None
    assert inference.engine is None


def test_engine_clear_cache(psf_test_setup):
    """Test that clear_cache resets the internal PSF cache."""
    inference = psf_test_setup["inference"]
    expected_psfs = psf_test_setup["expected_psfs"]

    # Create the engine and compute PSFs
    inference.engine = PSFInferenceEngine(
        trained_model=inference.trained_psf_model,
        batch_size=inference.batch_size,
        output_dim=inference.output_dim,
    )

    inference.engine._inferred_psfs = expected_psfs

    # Clear the cache
    inference.engine.clear_cache()

    # Check that the internal cache is None
    assert inference.engine._inferred_psfs is None, (
        "PSF cache should be cleared to None"
    )
