# test_train.py
import pytest
import tensorflow as tf
import numpy as np
from unittest.mock import patch, MagicMock, call
from wf_psf.training import train_utils

@pytest.fixture
def mock_model():
    # Create a mock model to simulate TensorFlow's `fit` method
    mock_model = MagicMock()
    mock_model.fit.return_value = MagicMock()  # Mock the returned history object
    return mock_model

@pytest.fixture
def mock_data():
    # Dummy data for training
    inputs = np.random.randn(10, 64, 64, 3)  # Example input data (10 samples, 64x64x3 images)
    outputs = np.random.randn(10, 64, 64, 1)  # Example output data (10 samples, 64x64x1 labels)
    return inputs, outputs


def test_train_cycle_part_without_sample_weights(mock_model, mock_data):
    inputs, outputs = mock_data
    # Test the training function when sample weights are not used
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]

    # Call the function (mocking the model's `fit` method)
    result = train_utils.train_cycle_part(
        mock_model,
        inputs,
        outputs,
        batch_size=2,
        epochs=1,
        validation_data=None,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        callbacks=None,
        sample_weight=None,
        verbose=1
    )

    # Assert that the `fit` method was called with the correct arguments
    mock_model.fit.assert_called_once_with(
        x=inputs,
        y=outputs,
        batch_size=2,
        epochs=1,
        validation_data=None,
        callbacks=None,
        sample_weight=None,
        verbose=1
    )
    assert result is not None

def test_train_cycle_part_with_sample_weights(mock_model, mock_data):
    inputs, outputs = mock_data
    # Test the training function when sample weights are used
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]

    # Mock sample weights
    sample_weights = np.ones(inputs.shape[0])  # Uniform sample weights

    result = train_utils.train_cycle_part(
        mock_model,
        inputs,
        outputs,
        batch_size=2,
        epochs=1,
        validation_data=None,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        callbacks=None,
        sample_weight=sample_weights, 
        verbose=1
    )

    # Assert that the `fit` method was called with non-None sample weights
    mock_model.fit.assert_called_once_with(
        x=inputs,
        y=outputs,
        batch_size=2,
        epochs=1,
        validation_data=None,
        callbacks=None,
        sample_weight=sample_weights,  
        verbose=1
    )
    assert result is not None


@pytest.mark.parametrize(
    "cycle_def, param_train, non_param_train, param_zero, non_param_zero",
    [
        ("parametric", True, False, False, False),
        ("non-parametric", False, True, False, False),
        ("complete", True, True, False, False),
        ("only-parametric", True, False, True, False),
        ("only-non-parametric", False, True, False, True),
    ],
)
def test_general_train_cycle_handles_cycle_def(cycle_def, param_train, non_param_train, param_zero, non_param_zero):
    # Mock the tf_semiparam_field
    mock_model = MagicMock()
    
    # Dummy inputs, outputs, and validation data
    inputs = tf.random.normal((10, 5))
    outputs = tf.random.normal((10, 1))
    validation_data = (tf.random.normal((5, 5)), tf.random.normal((5, 1)))

    # Call the function
    train_utils.general_train_cycle(
        tf_semiparam_field=mock_model,
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
    if param_train and non_param_train:
        # Ensure both parametric and non-parametric layers are set as trainable
        mock_model.set_trainable_layers.assert_has_calls([
            call(param_bool=True, nonparam_bool=False),  # Parametric training
            call(param_bool=False, nonparam_bool=True)   # Non-parametric training
        ], any_order=False)  # Ensures the calls happened in order
    elif param_train:
        # Only parametric layers should be trainable
        mock_model.set_trainable_layers.assert_called_once_with(param_bool=True, nonparam_bool=False)
    elif non_param_train:
        # Only non-parametric layers should be trainable
        mock_model.set_trainable_layers.assert_called_once_with(param_bool=False, nonparam_bool=True)


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


def test_general_train_cycle_with_callbacks(
    mock_model
):
    # Test setup with some mock parameters
    inputs = MagicMock()
    outputs = MagicMock()
    validation_data = MagicMock()
    batch_size = 32
    learning_rate_param = 1e-2
    learning_rate_non_param = 1.0
    n_epochs_param = 20
    n_epochs_non_param = 100
    param_optim = None
    non_param_optim = None
    param_loss = None
    non_param_loss = None
    param_metrics = None
    non_param_metrics = None
    param_callback = [MagicMock(name="param_callback1")] 
    non_param_callback = [MagicMock(name="non_param_callback1")] 
    general_callback = [MagicMock(name="general_callback1")] 
    first_run = False
    cycle_def = "complete"
    use_sample_weights = False
    verbose = 1

    # Create the mock callbacks once
    param_mock = MagicMock(name="param_callback1")
    general_mock = MagicMock(name="general_callback1")
    sample_weight = None

    # Combine the two callback lists directly as `get_callbacks` would
    parametric_callbacks = param_callback + general_callback
    non_parametric_callbacks = non_param_callback + general_callback

    # Mock the callbacks return value using patching
    with patch('wf_psf.training.train_utils.get_callbacks') as mock_get_callbacks, \
        patch('wf_psf.training.train_utils.train_cycle_part') as mock_train_cycle_part:
        
        # Use side_effect to return different callback lists depending on the cycle
        mock_get_callbacks.side_effect = [
        parametric_callbacks,
        non_parametric_callbacks
        ]

        # Mock train_cycle_part to check that callbacks are passed correctly
        mock_train_cycle_part.return_value = (mock_model, MagicMock())  # Mock return value of training

        # Call the function under test
        tf_semiparam_field, hist_param, hist_non_param = train_utils.general_train_cycle(
            mock_model,
            inputs,
            outputs,
            validation_data,
            batch_size,
            learning_rate_param,
            learning_rate_non_param,
            n_epochs_param,
            n_epochs_non_param,
            param_optim,
            non_param_optim,
            param_loss,
            non_param_loss,
            param_metrics,
            non_param_metrics,
            param_callback,
            non_param_callback,
            general_callback,
            first_run,
            cycle_def,
            use_sample_weights,
            verbose
        )

        # Check if get_callbacks was called with the correct arguments
        mock_get_callbacks.assert_any_call(param_callback, general_callback)
        mock_get_callbacks.assert_any_call(non_param_callback, general_callback)

        # Ensure the correct callbacks are returned by `get_callbacks` and used in the cycle
        assert tf_semiparam_field is not None
        assert hist_param is not None
        assert hist_non_param is not None

        # Check if mock_train_cycle_part was called as expected   
        assert mock_train_cycle_part.call_count == 2  

        print(mock_train_cycle_part.call_args_list)
        # First call
        first_call_args = mock_train_cycle_part.call_args_list[0][1]
        assert first_call_args["cycle_part"] == "parametric"
        assert first_call_args["epochs"] == 20
        assert first_call_args["callbacks"] == parametric_callbacks

        # Second call
        second_call_args = mock_train_cycle_part.call_args_list[1][1]
        assert second_call_args["cycle_part"] == "non-parametric"
        assert second_call_args["epochs"] == 100
        assert second_call_args["callbacks"] == non_parametric_callbacks


