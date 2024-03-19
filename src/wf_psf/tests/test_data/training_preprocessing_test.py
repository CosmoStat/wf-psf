import pytest
import os
import numpy as np
import tensorflow as tf
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.data.training_preprocessing import DataHandler, get_obs_positions
from wf_psf.psf_models import psf_models


def test_initialize_load_dataset(data_params, simPSF):
    # Test loading dataset without initialization
    data_handler = DataHandler(
        "train", data_params, simPSF, n_bins_lambda=10, init_flag=False
    )
    assert data_handler.dataset is None  # Dataset should not be loaded

    # Test loading dataset with initialization
    data_handler = DataHandler(
        "train", data_params, simPSF, n_bins_lambda=10, init_flag=True
    )
    assert data_handler.dataset is not None  # Dataset should be loaded


def test_initialize_process_sed_data(data_params, simPSF):
    # Test processing SED data without initialization
    data_handler = DataHandler(
        "train", data_params, simPSF, n_bins_lambda=10, init_flag=False
    )
    assert data_handler.sed_data is None  # SED data should not be processed

    # Test processing SED data with initialization
    data_handler = DataHandler(
        "train", data_params, simPSF, n_bins_lambda=10, init_flag=True
    )
    assert data_handler.sed_data is not None  # SED data should be processed


def test_load_train_dataset(tmp_path, data_params, simPSF):
    # Create a temporary directory and a temporary data file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    temp_data_dir = data_dir / "train_data.npy"

    # Mock dataset
    mock_dataset = {
        "positions": np.array([[1, 2], [3, 4]]),
        "noisy_stars": np.array([[5, 6], [7, 8]]),
        "SEDs": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
    }

    # Save the mock dataset to the temporary data file
    np.save(temp_data_dir, mock_dataset)

    # Initialize DataHandler instance
    data_params = RecursiveNamespace(
        train=RecursiveNamespace(data_dir=str(data_dir), file="train_data.npy")
    )

    n_bins_lambda = 10
    data_handler = DataHandler("train", data_params, simPSF, n_bins_lambda, False)

    # Call the load_dataset method
    data_handler.load_dataset()

    # Assertions
    assert np.array_equal(data_handler.dataset["positions"], mock_dataset["positions"])
    assert np.array_equal(
        data_handler.dataset["noisy_stars"], mock_dataset["noisy_stars"]
    )
    assert np.array_equal(data_handler.dataset["SEDs"], mock_dataset["SEDs"])


def test_load_test_dataset(tmp_path, data_params, simPSF):
    # Create a temporary directory and a temporary data file
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    temp_data_dir = data_dir / "test_data.npy"

    # Mock dataset
    mock_dataset = {
        "positions": np.array([[1, 2], [3, 4]]),
        "stars": np.array([[5, 6], [7, 8]]),
        "SEDs": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
    }

    # Save the mock dataset to the temporary data file
    np.save(temp_data_dir, mock_dataset)

    # Initialize DataHandler instance
    data_params = RecursiveNamespace(
        test=RecursiveNamespace(data_dir=str(data_dir), file="test_data.npy")
    )

    n_bins_lambda = 10
    data_handler = DataHandler("test", data_params, simPSF, n_bins_lambda, False)

    # Call the load_dataset method
    data_handler.load_dataset()

    # Assertions
    assert np.array_equal(data_handler.dataset["positions"], mock_dataset["positions"])
    assert np.array_equal(data_handler.dataset["stars"], mock_dataset["stars"])
    assert np.array_equal(data_handler.dataset["SEDs"], mock_dataset["SEDs"])


def test_process_sed_data(data_params, simPSF):
    mock_dataset = {
        "positions": np.array([[1, 2], [3, 4]]),
        "noisy_stars": np.array([[5, 6], [7, 8]]),
        "SEDs": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
    }
    # Initialize DataHandler instance
    n_bins_lambda = 4
    data_handler = DataHandler("train", data_params, simPSF, n_bins_lambda, False)

    data_handler.dataset = mock_dataset
    data_handler.process_sed_data()
    # Assertions
    assert isinstance(data_handler.sed_data, tf.Tensor)
    assert data_handler.sed_data.dtype == tf.float32
    assert data_handler.sed_data.shape == (
        len(data_handler.dataset["positions"]),
        n_bins_lambda,
        len(["feasible_N", "feasible_wv", "SED_norm"]),
    )


@pytest.fixture
def mock_data():
    class MockData:
        def __init__(self, training_positions, test_positions):
            self.training_data = MockDataset(training_positions)
            self.test_data = MockDataset(test_positions)

    class MockDataset:
        def __init__(self, positions):
            self.dataset = {"positions": positions}

    training_positions = np.array([[1, 2], [3, 4]])
    test_positions = np.array([[5, 6], [7, 8]])
    return MockData(training_positions, test_positions)
    
def test_get_obs_positions(mock_data):
    observed_positions = get_obs_positions(mock_data)
    expected_positions = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert tf.reduce_all(tf.equal(observed_positions, expected_positions))