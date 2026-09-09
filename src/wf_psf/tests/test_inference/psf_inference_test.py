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
from wf_psf.data.schemas import DatasetMode
from wf_psf.inference.psf_dataset import PSFDataset
from wf_psf.inference.psf_inference import (
    InferenceConfigHandler,
    PSFInference,
    PSFInferenceEngine,
    generate_psf_models,
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


@pytest.fixture
def mock_dataset(mock_inference_config):
    """
    Unified mock dataset fixture for positions, SEDs, and expected PSFs.
    Can be sliced or reshaped for single/multi-source tests.
    """
    tf_dataset = {}

    # Multi-source example
    tf_dataset["num_sources_multi"] = 2
    tf_dataset["num_bins"] = mock_inference_config.inference.model_params.n_bins_lambda
    tf_dataset["output_dim"] = mock_inference_config.inference.model_params.output_dim

    # PSF Dataset class
    psf_dataset_multi = PSFDataset(
        positions=np.array([[0.1, 0.1], [0.2, 0.2]], dtype=np.float32),
        seds=np.random.rand(
            tf_dataset["num_sources_multi"], tf_dataset["num_bins"], 2
        ).astype(np.float32),
        sources=np.random.rand(
            tf_dataset["num_sources_multi"],
            tf_dataset["output_dim"],
            tf_dataset["output_dim"],
        ).astype(np.float32),
    )

    # Single-source example (reshaped or sliced)
    psf_dataset_single = PSFDataset(
        positions=np.asarray(psf_dataset_multi.positions)[:1],
        seds=np.asarray(psf_dataset_multi.seds)[:1],
        sources=np.asarray(psf_dataset_multi.sources)[:1],
    )

    # Convert to tensors
    tf_dataset["positions_multi_tf"] = tf.convert_to_tensor(
        np.asarray(psf_dataset_multi.positions)
    )
    tf_dataset["seds_multi_tf"] = tf.convert_to_tensor(
        np.asarray(psf_dataset_multi.seds)
    )
    tf_dataset["expected_psfs_multi"] = psf_dataset_multi.sources

    tf_dataset["positions_single_tf"] = tf.convert_to_tensor(
        np.asarray(psf_dataset_single.positions)
    )
    tf_dataset["seds_single_tf"] = tf.convert_to_tensor(
        np.asarray(psf_dataset_single.seds)
    )
    tf_dataset["expected_psfs_single"] = psf_dataset_single.sources

    return psf_dataset_multi, psf_dataset_single, tf_dataset


@pytest.fixture
def mock_data_adapter():
    """
    Adapter fixture that uses the unified mock dataset.
    """
    adapter = MagicMock()
    adapter.representation_state = RepresentationState.NUMPY
    adapter.convert_to_tensorflow = MagicMock()
    return adapter


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
        psf_dataset, _, tf_dataset = mock_dataset
        num_sources = tf_dataset["num_sources_multi"]
        positions = tf_dataset["positions_multi_tf"]
        seds = tf_dataset["seds_multi_tf"]
        expected_psfs = tf_dataset["expected_psfs_multi"]
    else:
        _, psf_dataset, tf_dataset = mock_dataset
        num_sources = 1
        positions = tf_dataset["positions_single_tf"]
        seds = tf_dataset["seds_single_tf"]
        expected_psfs = tf_dataset["expected_psfs_single"]

    psf_generator = PSFInference(
        inference_config_path="dummy_path.yaml", dataset=psf_dataset
    )
    psf_generator._config_handler = MagicMock()
    psf_generator._config_handler.inference_config = mock_inference_config.inference
    psf_generator._trained_psf_model = MagicMock()

    return {
        "mock_psf_generator": psf_generator,
        "mock_positions": positions,
        "mock_seds": seds,
        "expected_psfs": expected_psfs,
        "num_sources": num_sources,
        "num_bins": tf_dataset["num_bins"],
        "output_dim": tf_dataset["output_dim"],
        "mode": request.param,  # "single" or "multi"
    }


@pytest.fixture
def mock_compute_psfs_with_cache(psf_setup):
    """
    Fixture that patches PSFInferenceEngine.compute_psfs with a side effect
    that populates the engine's cache.
    Works for both single-star and multi-star setups.
    """
    mock_psf_generator = psf_setup["mock_psf_generator"]
    mock_positions = psf_setup["mock_positions"]
    mock_seds = psf_setup["mock_seds"]
    expected_psfs = psf_setup["expected_psfs"]

    with patch.object(PSFInferenceEngine, "compute_psfs") as mock_compute_psfs:

        def fake_compute_psfs(positions, seds):
            # Populate the engine cache with the expected PSFs
            mock_psf_generator.engine._inferred_psfs = expected_psfs
            return expected_psfs

        mock_compute_psfs.side_effect = fake_compute_psfs

        yield {
            "mock": mock_compute_psfs,
            "psf_generator": mock_psf_generator,
            "positions": mock_positions,
            "seds": mock_seds,
            "expected_psfs": expected_psfs,
            "mode": psf_setup["mode"],  # "single" or "multi"
        }


# -----------------------
# Tests
# -----------------------
def test_prepare_configs(mock_training_config, mock_inference_config, mock_dataset):
    """Test preparing configurations for inference."""
    # Mock the model_params object with some initial values
    training_config = mock_training_config
    inference_config = mock_inference_config

    # Extract PSF dataset from mock_daaset
    psf_dataset, _, _ = mock_dataset

    # Make copy of the original training config model_params
    original_model_params = mock_training_config.training.model_params

    # Instantiate PSFInference
    psf_generator = PSFInference(
        inference_config_path="/dummy/path.yaml", dataset=psf_dataset
    )

    # Mock the config handler attribute with a mock InferenceConfigHandler
    mock_config_handler = MagicMock(spec=InferenceConfigHandler)
    mock_config_handler.training_config = training_config.training
    mock_config_handler.inference_config = inference_config.inference

    # Patch the overwrite_model_params to use the real static method
    mock_config_handler.overwrite_model_params.side_effect = (
        InferenceConfigHandler.overwrite_model_params
    )

    psf_generator._config_handler = mock_config_handler

    # Run prepare_configs
    psf_generator.prepare_configs()

    # Assert that the training model_params were updated
    assert original_model_params.output_Q == 1
    assert original_model_params.output_dim == 64


def test_batch_size_positive(mock_dataset):
    psf_dataset, _, _ = mock_dataset
    inference = PSFInference(
        inference_config_path="dummy_path.yaml", dataset=psf_dataset
    )
    inference._config_handler = MagicMock()
    inference._config_handler.inference_config = SimpleNamespace(
        batch_size=4, model_params=SimpleNamespace(output_dim=32)
    )
    assert inference.batch_size == 4


@pytest.mark.parametrize(("schema_mode"), ["INFERENCE", "EVALUATION"])
def test_schema_mode(mock_inference_config, schema_mode):
    handler = InferenceConfigHandler.__new__(InferenceConfigHandler)
    mock_inference_config.inference.schema_mode = schema_mode
    handler.inference_config = mock_inference_config.inference

    assert handler.schema_mode == DatasetMode[schema_mode]


def test_schema_mode_invalid(mock_inference_config):
    handler = InferenceConfigHandler.__new__(InferenceConfigHandler)
    handler.inference_config = mock_inference_config.inference
    handler.inference_config.schema_mode = "invalid"

    with pytest.raises(ValueError, match="Invalid dataset schema mode"):
        _ = handler.schema_mode


def test_compute_psfs_valid_inputs(psf_setup):
    psf_generator = psf_setup["mock_psf_generator"]
    expected_psfs = psf_setup["expected_psfs"]

    engine = PSFInferenceEngine(
        trained_model=psf_generator.trained_psf_model,
        batch_size=psf_generator.batch_size,
        output_dim=psf_generator.output_dim,
    )

    engine.trained_model.return_value = tf.convert_to_tensor(expected_psfs)

    tf_positions = psf_setup["mock_positions"]
    tf_seds = psf_setup["mock_seds"]

    inferred_psfs = engine.compute_psfs(positions=tf_positions, sed_data=tf_seds)

    np.testing.assert_array_equal(inferred_psfs, expected_psfs)


def test_compute_psfs_invalid_positions_shape(psf_setup):
    psf_generator = psf_setup["mock_psf_generator"]

    engine = PSFInferenceEngine(
        trained_model=psf_generator.trained_psf_model,
        batch_size=psf_generator.batch_size,
        output_dim=psf_generator.output_dim,
    )

    invalid_positions = tf.zeros((2, 3))
    sed_data = psf_setup["mock_seds"]

    with pytest.raises(
        ValueError,
        match=r"positions must have shape \(n_samples, 2\)",
    ):
        engine.compute_psfs(
            positions=invalid_positions,
            sed_data=sed_data,
        )


def test_compute_psfs_empty_positions(psf_setup):
    psf_generator = psf_setup["mock_psf_generator"]

    engine = PSFInferenceEngine(
        trained_model=psf_generator.trained_psf_model,
        batch_size=psf_generator.batch_size,
        output_dim=psf_generator.output_dim,
    )

    empty_positions = tf.zeros((0, 2))
    empty_seds = tf.zeros((0, 10, 2))

    with pytest.raises(
        ValueError,
        match="positions must contain at least one sample",
    ):
        engine.compute_psfs(
            positions=empty_positions,
            sed_data=empty_seds,
        )


@patch("wf_psf.inference.psf_inference.DataAdapterFactory.build")
@patch("wf_psf.inference.psf_inference.psf_models.simPSF")
def test_data_adapter_property_adapter_build(
    _, mock_build, psf_setup, mock_data_adapter
):
    psf_generator = psf_setup["mock_psf_generator"]

    mock_build.return_value = mock_data_adapter

    adapter = psf_generator.inference_data_adapter

    assert adapter == mock_data_adapter
    mock_build.assert_called_once()


@patch("wf_psf.inference.psf_inference.DataAdapterFactory.build")
@patch("wf_psf.inference.psf_inference.psf_models.simPSF")
def test_data_adapter_cached(_, mock_build, psf_setup, mock_data_adapter):
    psf_generator = psf_setup["mock_psf_generator"]
    mock_build.return_value = mock_data_adapter

    adapter1 = psf_generator.inference_data_adapter
    adapter2 = psf_generator.inference_data_adapter

    assert adapter1 is adapter2
    mock_build.assert_called_once()


@patch("wf_psf.inference.psf_inference.DataAdapterFactory.build")
def test_data_adapter_no_conversion_if_tensorflow(
    mock_build, psf_setup, mock_data_adapter
):
    # Set representation state to TENSORFLOW
    mock_data_adapter.representation_state = RepresentationState.TENSORFLOW

    mock_build.return_value = mock_data_adapter

    psf_generator = psf_setup["mock_psf_generator"]

    psf_generator.inference_data_adapter

    mock_data_adapter.convert_to_tensorflow.assert_not_called()


@patch("wf_psf.inference.psf_inference.load_trained_psf_model")
def test_load_inference_model(
    mock_load_trained_psf_model,
    mock_training_config,
    mock_inference_config,
    mock_dataset,
    mock_data_adapter,
):
    psf_dataset, _, _ = mock_dataset

    psf_inf = PSFInference(inference_config_path="dummy_path.yaml", dataset=psf_dataset)

    mock_config_handler = MagicMock(spec=InferenceConfigHandler)
    mock_config_handler.trained_model_path = "mock/path/to/model"
    mock_config_handler.training_config = mock_training_config.training
    mock_config_handler.inference_config = mock_inference_config.inference
    mock_config_handler.model_subdir = "psf_model"

    psf_inf._config_handler = mock_config_handler
    psf_inf._model_data_adapter = mock_data_adapter

    psf_inf.load_inference_model()

    mock_load_trained_psf_model.assert_called_once()


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
    psf_setup,
):
    # Mock factory build
    mock_build.return_value = mock_data_adapter
    psf_generator = psf_setup["mock_psf_generator"]

    # Set complete_data in mock_data_adapter
    mock_data_adapter.complete_data = {
        "positions": psf_setup["mock_positions"],
        "seds": psf_setup["mock_seds"],
    }

    # Lazy-load inference.data_adapter
    psf_generator.inference_data_adapter

    mock_positions = mock_data_adapter.complete_data["positions"]
    mock_seds = mock_data_adapter.complete_data["seds"]
    expected_psfs = psf_setup["expected_psfs"]

    mock_compute_psfs.return_value = expected_psfs

    psfs = psf_generator.run_inference()

    assert isinstance(psfs, np.ndarray)
    assert psfs.shape == expected_psfs.shape
    mock_prepare_configs.assert_called_once()
    mock_compute_psfs.assert_called_once_with(mock_positions, mock_seds)
    mock_data_adapter.convert_to_tensorflow.assert_called_once_with(
        psf_generator.simPSF,
        psf_generator.n_bins_lambda,
        mode=psf_generator.config_handler.schema_mode,
    )


@patch.object(PSFInference, "prepare_configs")
@patch("wf_psf.inference.psf_inference.DataAdapterFactory.build")
@patch("wf_psf.inference.psf_inference.psf_models.simPSF")
def test_run_inference_invalid_complete_data_type(
    _,
    mock_build,
    mock_prepare_configs,
    mock_data_adapter,
    psf_setup,
):
    # Mock factory build
    mock_build.return_value = mock_data_adapter
    psf_generator = psf_setup["mock_psf_generator"]

    # Set complete_data in mock_data_adapter
    mock_data_adapter.complete_data = []

    with pytest.raises(
        TypeError,
        match="Expected inference adapter complete_data to be a dict",
    ):
        psf_generator.run_inference()


@pytest.mark.parametrize(
    ("invalid_field", "error_message"),
    [
        (
            "positions",
            "Expected inference positions to be a TensorFlow Tensor",
        ),
        (
            "seds",
            "Expected inference SED data to be a TensorFlow Tensor",
        ),
    ],
)
@patch.object(PSFInference, "prepare_configs")
@patch("wf_psf.inference.psf_inference.DataAdapterFactory.build")
@patch("wf_psf.inference.psf_inference.psf_models.simPSF")
def test_run_inference_invalid_product_type(
    _,
    mock_build,
    mock_prepare_configs,
    mock_data_adapter,
    psf_setup,
    invalid_field,
    error_message,
):
    mock_build.return_value = mock_data_adapter
    psf_generator = psf_setup["mock_psf_generator"]

    complete_data = {
        "positions": psf_setup["mock_positions"],
        "seds": psf_setup["mock_seds"],
    }
    complete_data[invalid_field] = np.asarray(complete_data[invalid_field])

    mock_data_adapter.complete_data = complete_data

    with pytest.raises(TypeError, match=error_message):
        psf_generator.run_inference()


@patch("wf_psf.inference.psf_inference.psf_models.simPSF")
def test_simpsf_uses_updated_model_params(
    mock_simpsf, mock_training_config, mock_inference_config, mock_dataset
):
    """Test that simPSF uses the updated model parameters."""
    training_config = mock_training_config.training
    inference_config = mock_inference_config.inference
    psf_dataset, _, _ = mock_dataset

    # Set the expected output_Q
    expected_output_Q = inference_config.model_params.output_Q
    training_config.model_params.output_Q = expected_output_Q

    # Create fake sim psf instance
    fake_psf_instance = MagicMock()
    fake_psf_instance.output_Q = expected_output_Q
    mock_simpsf.return_value = fake_psf_instance

    mock_config_handler = MagicMock(spec=InferenceConfigHandler)
    mock_config_handler.trained_model_path = "mock/path/to/model"
    mock_config_handler.training_config = training_config
    mock_config_handler.inference_config = inference_config
    mock_config_handler.model_subdir = "psf_model"
    mock_config_handler.data_config = MagicMock()

    modeller = PSFInference(
        inference_config_path="dummy_path.yaml", dataset=psf_dataset
    )
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
    psf_generator = mock_compute_psfs_with_cache["psf_generator"]
    expected_psfs = mock_compute_psfs_with_cache["expected_psfs"]

    psf_generator.engine = MagicMock()
    psf_generator.engine.inferred_psfs = None
    psf_generator.engine.get_psfs.return_value = expected_psfs

    with patch.object(psf_generator, "run_inference") as mock_run:

        def fake_run():
            psf_generator.engine.inferred_psfs = expected_psfs
            return expected_psfs

        mock_run.side_effect = fake_run

        psfs_1 = psf_generator.get_psfs()
        psfs_2 = psf_generator.get_psfs()

        np.testing.assert_array_equal(psfs_1, expected_psfs)
        np.testing.assert_array_equal(psfs_2, expected_psfs)

        mock_run.assert_called_once()


def test_inference_clear_cache(psf_setup):
    """Test that PSFInference clear_cache resets the instance of PSFInference."""
    inference = psf_setup["mock_psf_generator"]
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


def test_engine_clear_cache(psf_setup):
    """Test that clear_cache resets the internal PSF cache."""
    inference = psf_setup["mock_psf_generator"]
    expected_psfs = psf_setup["expected_psfs"]

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


@patch("wf_psf.inference.psf_inference.PSFInference")
def test_generate_psf_models(mock_psf_inference, mock_dataset):
    expected_psfs = np.random.rand(2, 32, 32)
    mock_psf_inference.return_value.get_psfs.return_value = expected_psfs

    dataset, _, _ = mock_dataset

    result = generate_psf_models(
        dataset=dataset,
        inference_config_path="dummy_path.yaml",
    )

    mock_psf_inference.assert_called_once_with(
        inference_config_path="dummy_path.yaml",
        dataset=dataset,
    )
    mock_psf_inference.return_value.get_psfs.assert_called_once_with()

    np.testing.assert_array_equal(result, expected_psfs)
