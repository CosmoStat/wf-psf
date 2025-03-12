# test_train_utils.py
import pytest
import tensorflow as tf
import numpy as np
from unittest.mock import patch, MagicMock, call
from wf_psf.training import train_utils


@pytest.fixture
def mock_data():
    # Dummy data for training
    inputs = np.random.randn(10, 64, 64, 3)  # Example input data (10 samples, 64x64x3 images)
    outputs = np.random.randn(10, 64, 64, 1)  # Example output data (10 samples, 64x64x1 labels)
    return inputs, outputs

@pytest.fixture
def mock_model():
    # Create a mock model to simulate TensorFlow's `fit` method
    mock_model = MagicMock()
    mock_model.fit.return_value = MagicMock()  # Mock the returned history object
    return mock_model

@pytest.fixture
def mock_noise_estimator():
    """Mock the NoiseEstimator class to return a fixed noise estimate."""
    with patch("wf_psf.training.train_utils.NoiseEstimator") as MockNoiseEstimator:
        mock_instance = MockNoiseEstimator.return_value
        mock_instance.estimate_noise.side_effect = lambda img: np.std(img)  # Mock behavior
        yield mock_instance

@pytest.fixture
def mock_test_setup(mock_model, mock_data):
    """Fixture setting up common test parameters and including mock model and data."""
    inputs, outputs = mock_data
    return {
        "mock_model": mock_model,
        "inputs": inputs,
        "outputs": outputs,
        "validation_data": MagicMock(),
        "batch_size": 32,
        "learning_rate_param": 1e-2,
        "learning_rate_non_param": 1.0,
        "n_epochs_param": 20,
        "n_epochs_non_param": 100,
        "param_optim": None,
        "non_param_optim": None,
        "param_loss": None,
        "non_param_loss": None,
        "param_metrics": None,
        "non_param_metrics": None,
        "first_run": False,
        "use_sample_weights": False,
        "verbose": 1
    }

@pytest.mark.parametrize("use_sample_weights, expected_output", [
    (False, None),
    (True, np.ndarray),  # When enabled, it should return an array
])
def test_calculate_sample_weights_no_weights(mock_noise_estimator, use_sample_weights, expected_output):
    """Test when sample weights are disabled and ensure correct return type."""
    outputs = np.random.rand(5, 32, 32)  # (batch_size=5, height=32, width=32)
    result = train_utils.calculate_sample_weights(outputs, use_sample_weights)
    if not use_sample_weights:
        assert result is None
    else:
        assert isinstance(result, expected_output)

def test_calculate_sample_weights(mock_noise_estimator):
    """Test sample weighting strategy with random images."""
    outputs = np.random.rand(10, 32, 32)  # 10 images of size 32x32
    result = train_utils.calculate_sample_weights(outputs, use_sample_weights=True)

    # Check the type and shape of the result
    assert isinstance(result, np.ndarray)
    assert result.shape == (10,)  # Should return weights for each image

    # Check that the weights are positive
    assert np.all(result > 0)

    # Check that the median of the weights is around 1
    assert np.isclose(np.median(result), 1, atol=0.1)

def test_calculate_sample_weights_small_variance(mock_noise_estimator):
    """Test edge case where images have very small variance (almost constant images with some noise)."""
    outputs = np.random.normal(loc=0.0, scale=1e-6, size=(5, 32, 32))  # Tiny noise
    weights = train_utils.calculate_sample_weights(outputs, use_sample_weights=True)
    
    # Use np.allclose to check if the weights are nearly identical within a tolerance.
    assert np.allclose(weights, weights[0], atol=1e-1)

    # Check that weights are not too far apart
    assert np.max(weights) - np.min(weights) < 0.1, f"Weight difference too large: {weights}"

    # Check that the weights are non-negative and reasonable
    assert np.all(weights >= 0), f"Weights are negative: {weights}"

    # Check if the weight values seem appropriate (within a reasonable range, close to 1).
    assert np.all(weights <= 2), f"Weights exceed the expected upper bound: {weights}"

    # Check that the median of weights is close to 1
    assert np.isclose(np.median(weights), 1, atol=0.05), f"Median is not close to 1: {np.median(weights)}"

def test_calculate_sample_weights_high_variance(mock_noise_estimator):
    """Test case for high variance (noisy images)."""
    # Create high variance images with more noise
    outputs = np.random.normal(loc=0.0, scale=10.0, size=(5, 32, 32))  # Larger noise

    # Calculate sample weights
    weights = train_utils.calculate_sample_weights(outputs, use_sample_weights=True)
    
    # Check that weights are relatively lower for high variance images
    # Compare with weights from a small variance case
    small_variance_outputs = np.random.normal(loc=0.0, scale=1e-6, size=(5, 32, 32))  # Small noise
    small_variance_weights = train_utils.calculate_sample_weights(small_variance_outputs, use_sample_weights=True)
    
    # Check if the median of the high variance weights is smaller than the median of the low variance weights
    assert np.median(weights) <= np.median(small_variance_weights), (
        f"High variance weights' median should be smaller than low variance weights' median: "
        f"{np.median(weights)} vs {np.median(small_variance_weights)}"
    )

    # Optionally check if weights are within a reasonable range (non-negative, etc.)
    assert np.all(weights >= 0), f"Weights are negative: {weights}"


@pytest.mark.parametrize("sample_weight", [None, np.random.randn(10)])
def test_train_cycle_part(mock_test_setup, sample_weight):
    mock_model = mock_test_setup["mock_model"]
    inputs = mock_test_setup["inputs"]
    outputs = mock_test_setup["outputs"]
    batch_size = 2
    epochs = 1
    validation_data = None
    sample_weight = sample_weight
    verbose = mock_test_setup["verbose"]

    # Test the training function when sample weights are not used
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]

    # Call the function (mocking the model's `fit` method)
    result = train_utils.train_cycle_part(
        psf_model=mock_model,
        inputs=inputs,
        outputs=outputs,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        callbacks=None,
        sample_weight=sample_weight,
        verbose=verbose
    )

    # Assert that the `fit` method was called with the correct arguments
    mock_model.fit.assert_called_once_with(
        x=inputs,
        y=outputs,
        batch_size=2,
        epochs=1,
        validation_data=None,
        callbacks=None,
        sample_weight=sample_weight,
        verbose=1
    )
    assert result is not None


@pytest.mark.parametrize(
    "cycle_def, param_train, non_param_train, param_zero, non_param_zero",
    [
        pytest.param("parametric", True, False, False, False, id="train-parametric"),
        pytest.param("non-parametric", False, True, False, False, id="train-non-parametric"),
        pytest.param("complete", True, True, False, False, id="train-complete"),
        pytest.param("only-parametric", True, False, True, False, id="train-only-parametric"),
       pytest.param("only-non-parametric", False, True, False, True, id="train-only-non-parametric"),
    ],
)
def test_general_train_cycle_handles_cycle_def(mock_test_setup, cycle_def, param_train, non_param_train, param_zero, non_param_zero):
    # Unpack test setup
    mock_model = mock_test_setup["mock_model"]
    
    # Dummy inputs, outputs, and validation data
    inputs = tf.random.normal((10, 5))
    outputs = tf.random.normal((10, 1))
    validation_data = (tf.random.normal((5, 5)), tf.random.normal((5, 1)))

    # Call the function
    train_utils.general_train_cycle(
        psf_model=mock_model,
        inputs=inputs,
        outputs=outputs,
        validation_data=validation_data,
        batch_size=2,
        learning_rate_param=0.01,
        learning_rate_non_param=1.0,
        n_epochs_param=10, 
        n_epochs_non_param=20,
        cycle_def=cycle_def,
    )

    # Assertions for setting parts to zero
    if param_zero:
        mock_model.set_zero_nonparam.assert_called_once()
    else:
        mock_model.set_zero_nonparam.assert_not_called()
    
    if non_param_zero:
        mock_model.get_coeff_matrix.assert_called_once()
        mock_model.assign_coeff_matrix.assert_called_once()
    else:
        mock_model.get_coeff_matrix.assert_not_called()
        mock_model.assign_coeff_matrix.assert_not_called()

    # Assertions for training layers
    expected_calls = []
    if param_train:
        expected_calls.append(call(param_bool=True, nonparam_bool=False))
    if non_param_train:
        expected_calls.append(call(param_bool=False, nonparam_bool=True))

    mock_model.set_trainable_layers.assert_has_calls(expected_calls, any_order=False)


def test_get_callbacks():
    # Test when both are None
    assert train_utils.get_callbacks(None, None) is None

    # Test when one is None
    assert train_utils.get_callbacks(None, ['callback1']) == ['callback1']
    assert train_utils.get_callbacks(['callback1'], None) == ['callback1']

    # Test when both have values
    assert train_utils.get_callbacks(['callback1'], ['callback2']) == ['callback1', 'callback2']

    # Test when one is an empty list
    assert train_utils.get_callbacks([], ['callback2']) == ['callback2']
    assert train_utils.get_callbacks(['callback1'], []) == ['callback1']

@pytest.mark.parametrize(
    "cycle_def, param_callback, non_param_callback, general_callback",
    [
        ("complete", [MagicMock(name="param_callback1")], [MagicMock(name="non_param_callback1")], [MagicMock(name="general_callback1")]),
        ("parametric", [MagicMock(name="param_callback2")], None, [MagicMock(name="general_callback2")]),
        ("non-parametric", None, [MagicMock(name="non_param_callback2")], [MagicMock(name="general_callback2")]),
    ]
)
def test_general_train_cycle_with_callbacks(
    mock_test_setup, cycle_def, param_callback, non_param_callback, general_callback
):
    """Test general_train_cycle with different cycle_def and callback configurations."""

    # Unpack test setup
    mock_model = mock_test_setup["mock_model"]

    # Expected callback lists
    parametric_callbacks = (param_callback or []) + general_callback if cycle_def in ["complete", "parametric"] else None
    non_parametric_callbacks = (non_param_callback or []) + general_callback if cycle_def in ["complete", "non-parametric"] else None

    with patch("wf_psf.training.train_utils.get_callbacks") as mock_get_callbacks, \
         patch("wf_psf.training.train_utils.train_cycle_part") as mock_train_cycle_part:
        
        # Define side effect behavior dynamically
        callback_side_effects = []
        if parametric_callbacks:
            callback_side_effects.append(parametric_callbacks)
        if non_parametric_callbacks:
            callback_side_effects.append(non_parametric_callbacks)
        
        mock_get_callbacks.side_effect = callback_side_effects
        mock_train_cycle_part.return_value = (mock_model, MagicMock())

        # Call function under test
        psf_model, hist_param, hist_non_param = train_utils.general_train_cycle(
            mock_model,
            mock_test_setup["inputs"],
            mock_test_setup["outputs"],
            mock_test_setup["validation_data"],
            mock_test_setup["batch_size"],
            mock_test_setup["learning_rate_param"],
            mock_test_setup["learning_rate_non_param"],
            mock_test_setup["n_epochs_param"],
            mock_test_setup["n_epochs_non_param"],
            mock_test_setup["param_optim"],
            mock_test_setup["non_param_optim"],
            mock_test_setup["param_loss"],
            mock_test_setup["non_param_loss"],
            mock_test_setup["param_metrics"],
            mock_test_setup["non_param_metrics"],
            param_callback,
            non_param_callback,
            general_callback,
            mock_test_setup["first_run"],
            cycle_def,
            mock_test_setup["use_sample_weights"],
            mock_test_setup["verbose"]
        )

        # Validate calls to get_callbacks
        expected_callback_calls = [
            (param_callback, general_callback) if param_callback else None,
            (non_param_callback, general_callback) if non_param_callback else None
        ]
        for call in expected_callback_calls:
            if call:
                mock_get_callbacks.assert_any_call(*call)

        # Assertions based on cycle_def
        assert psf_model is not None
        if cycle_def in ("complete", "parametric"):
            assert hist_param is not None
        if cycle_def in ("complete", "non-parametric"):
            assert hist_non_param is not None

        # Ensure correct number of train_cycle_part calls
        assert mock_train_cycle_part.call_count == len(callback_side_effects)

    # Validate train_cycle_part calls
        expected_calls = []
        if parametric_callbacks:
            expected_calls.append(("parametric", 20, parametric_callbacks))
        if non_parametric_callbacks:
            expected_calls.append(("non-parametric", 100, non_parametric_callbacks))

        for idx, (cycle_part, epochs, callbacks) in enumerate(expected_calls):
            call_args = mock_train_cycle_part.call_args_list[idx][1]
            assert call_args["cycle_part"] == cycle_part
            assert call_args["epochs"] == epochs
            assert call_args["callbacks"] == callbacks