import pytest
import numpy as np
import tensorflow as tf
from wf_psf.data.data_adapter import DataAdapter
from wf_psf.data.data_utils import DatasetContainer
from wf_psf.training.training_data_adapter import TrainingDataAdapter

# ---- Fixtures ----


@pytest.fixture
def dummy_train_data():
    return {
        "positions": np.random.rand(5, 2),
        "sources": np.random.rand(5, 16, 16),
        "masks": np.ones((5, 16, 16)),
        "seds": np.random.rand(5, 3, 2),
    }


@pytest.fixture
def dummy_test_data():
    return {
        "positions": np.random.rand(3, 2),
        "sources": np.random.rand(3, 16, 16),
        "masks": np.ones((3, 16, 16)),
        "seds": np.random.rand(3, 3, 2),
    }


@pytest.fixture
def base_adapter(mocker, dummy_train_data, dummy_test_data):
    """Mock DataAdapter with train/test DatasetContainers."""
    adapter = mocker.Mock(spec=DataAdapter)
    adapter.train_data = DatasetContainer(dummy_train_data)
    adapter.test_data = DatasetContainer(dummy_test_data)
    return adapter


# ---- Tests ----
def test_train_inputs_returns_positions_and_seds(base_adapter):
    """Test train inputs contain positions and seds."""
    tda = TrainingDataAdapter(base_adapter)
    inputs = tda.train_inputs
    assert isinstance(inputs, list)
    assert len(inputs) == 2
    np.testing.assert_array_equal(inputs[0], base_adapter.train_data["positions"])
    np.testing.assert_array_equal(inputs[1], base_adapter.train_data["seds"])


def test_validation_inputs_returns_positions_and_seds(base_adapter):
    """Test validation inputs contain positions and seds."""
    tda = TrainingDataAdapter(base_adapter)
    inputs = tda.validation_inputs
    assert isinstance(inputs, list)
    assert len(inputs) == 2
    np.testing.assert_array_equal(inputs[0], base_adapter.test_data["positions"])
    np.testing.assert_array_equal(inputs[1], base_adapter.test_data["seds"])


def test_train_targets_mse(base_adapter):
    """Test train targets contains sources for mse loss."""
    tda = TrainingDataAdapter(base_adapter, loss_type="mse")
    targets = tda.train_targets
    np.testing.assert_array_equal(targets, base_adapter.train_data["sources"])


def test_validation_targets_mse(base_adapter):
    """Test validation targets contains sources for mse loss."""
    tda = TrainingDataAdapter(base_adapter, loss_type="mse")
    targets = tda.validation_targets
    np.testing.assert_array_equal(targets, base_adapter.test_data["sources"])


def test_train_targets_mask_mse(base_adapter):
    """Test sources and masks are packed for mask_mse loss."""
    tda = TrainingDataAdapter(base_adapter, loss_type="mask_mse")
    targets = tda.train_targets
    assert isinstance(targets, tf.Tensor)
    # shape check: last dimension stacked
    assert targets.shape[-1] == 2
    np.testing.assert_array_equal(targets[..., 0], base_adapter.train_data["sources"])
    np.testing.assert_array_equal(targets[..., 1], base_adapter.train_data["masks"])


def test_validation_targets_mask_mse(base_adapter):
    tda = TrainingDataAdapter(base_adapter, loss_type="mask_mse")
    targets = tda.validation_targets
    assert isinstance(targets, tf.Tensor)
    assert targets.shape[-1] == 2
    np.testing.assert_array_equal(targets[..., 0], base_adapter.test_data["sources"])
    np.testing.assert_array_equal(targets[..., 1], base_adapter.test_data["masks"])


def test_mask_mse_raises_if_masks_missing(base_adapter):
    # remove masks from train/test
    base_adapter.train_data.pop("masks")
    base_adapter.test_data.pop("masks")
    tda = TrainingDataAdapter(base_adapter, loss_type="mask_mse")
    with pytest.raises(ValueError, match="mask_mse requires masks for training"):
        _ = tda.train_targets
    with pytest.raises(ValueError, match="mask_mse requires masks for validation"):
        _ = tda.validation_targets


def test_inputs_without_seds(base_adapter):
    base_adapter.train_data.pop("seds")
    base_adapter.test_data.pop("seds")
    tda = TrainingDataAdapter(base_adapter)
    train_inputs = tda.train_inputs
    val_inputs = tda.validation_inputs
    # seds should be omitted if missing
    assert len(train_inputs) == 1
    assert np.array_equal(train_inputs[0], base_adapter.train_data["positions"])
    assert len(val_inputs) == 1
    assert np.array_equal(val_inputs[0], base_adapter.test_data["positions"])
