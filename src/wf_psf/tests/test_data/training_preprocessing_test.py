import pytest
import numpy as np
import tensorflow as tf
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.data.training_preprocessing import (
    DataHandler,
    get_obs_positions,
    get_zernike_prior,
    extract_star_data,
    compute_centroid_correction,
)
import logging
from unittest.mock import patch

class MockData:
    def __init__(
        self,
        training_positions,
        test_positions,
        training_zernike_priors,
        test_zernike_priors,
        noisy_stars=None,
        noisy_masks=None,
        stars=None,
        masks=None,
    ):
        self.training_data = MockDataset(
            positions=training_positions, 
            zernike_priors=training_zernike_priors,
            star_type="noisy_stars",
            stars=noisy_stars,
            masks=noisy_masks)
        self.test_data = MockDataset(
            positions=test_positions, 
            zernike_priors=test_zernike_priors,
            star_type="stars",
            stars=stars,
            masks=masks)


class MockDataset:
    def __init__(self, positions, zernike_priors, star_type, stars, masks):
        self.dataset = {"positions": positions, "zernike_prior": zernike_priors, star_type: stars, "masks": masks}


@pytest.fixture
def mock_data():
    # Mock data for testing
    # Mock training and test positions and Zernike priors
    training_positions = np.array([[1, 2], [3, 4]])
    test_positions = np.array([[5, 6], [7, 8]])
    training_zernike_priors = np.array([[0.1, 0.2], [0.3, 0.4]])
    test_zernike_priors = np.array([[0.5, 0.6], [0.7, 0.8]])
    # Mock noisy stars, stars and masks
    noisy_stars = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    noisy_masks = tf.constant([[1], [0]], dtype=tf.float32)
    stars = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    masks = tf.constant([[0], [1]], dtype=tf.float32)
     
    return MockData(
        training_positions, test_positions, training_zernike_priors, test_zernike_priors, noisy_stars, noisy_masks, stars, masks
    )


def test_process_sed_data(data_params, simPSF):
    # Test processing SED data without initialization
    data_handler = DataHandler(
        "train", data_params, simPSF, n_bins_lambda=10, load_data=False
    )
    assert data_handler.sed_data is None  # SED data should not be processed

    # Test processing SED data with initialization
    data_handler = DataHandler(
        "train", data_params, simPSF, n_bins_lambda=10, load_data=True
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
    data_handler = DataHandler("train", data_params, simPSF, n_bins_lambda, load_data=False)

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
    data_handler = DataHandler("test", data_params, simPSF, n_bins_lambda, load_data=False)

    # Call the load_dataset method
    data_handler.load_dataset()

    # Assertions
    assert np.array_equal(data_handler.dataset["positions"], mock_dataset["positions"])
    assert np.array_equal(data_handler.dataset["stars"], mock_dataset["stars"])
    assert np.array_equal(data_handler.dataset["SEDs"], mock_dataset["SEDs"])


def test_load_train_dataset_missing_noisy_stars(tmp_path, data_params, simPSF):
    """Test that a warning is raised if 'noisy_stars' is missing in training data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    temp_data_file = data_dir / "train_data.npy"

    mock_dataset = {
        "positions": np.array([[1, 2], [3, 4]]),  # No 'noisy_stars' key
        "SEDs": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
    }
    
    np.save(temp_data_file, mock_dataset)

    data_params = RecursiveNamespace(
        train=RecursiveNamespace(data_dir=str(data_dir), file="train_data.npy")
    )

    n_bins_lambda = 10
    data_handler = DataHandler("train", data_params, simPSF, n_bins_lambda, load_data=False)

    with patch("wf_psf.data.training_preprocessing.logger.warning") as mock_warning:
        data_handler.load_dataset()
        mock_warning.assert_called_with("Missing 'noisy_stars' in train dataset.")

def test_load_test_dataset_missing_stars(tmp_path, data_params, simPSF):
    """Test that a warning is raised if 'stars' is missing in test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    temp_data_file = data_dir / "test_data.npy"

    mock_dataset = {
        "positions": np.array([[1, 2], [3, 4]]),  # No 'stars' key
        "SEDs": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
    }

    np.save(temp_data_file, mock_dataset)

    data_params = RecursiveNamespace(
        test=RecursiveNamespace(data_dir=str(data_dir), file="test_data.npy")
    )

    n_bins_lambda = 10
    data_handler = DataHandler("test", data_params, simPSF, n_bins_lambda, load_data=False)

    with patch("wf_psf.data.training_preprocessing.logger.warning") as mock_warning:
        data_handler.load_dataset()
        mock_warning.assert_called_with("Missing 'stars' in test dataset.")


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


def test_get_obs_positions(mock_data):
    observed_positions = get_obs_positions(mock_data)
    expected_positions = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert tf.reduce_all(tf.equal(observed_positions, expected_positions))


def test_get_zernike_prior(model_params, mock_data):
    zernike_priors = get_zernike_prior(model_params, mock_data)
    expected_shape = (
        4,
        2,
    )  # Assuming 2 Zernike priors for each dataset (training and test)
    assert zernike_priors.shape == expected_shape


def test_get_zernike_prior_dtype(model_params, mock_data):
    zernike_priors = get_zernike_prior(model_params, mock_data)
    assert zernike_priors.dtype == np.float32


def test_get_zernike_prior_concatenation(model_params, mock_data):
    zernike_priors = get_zernike_prior(model_params, mock_data)
    expected_zernike_priors = tf.convert_to_tensor(
        np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]), dtype=tf.float32
    )

    assert np.array_equal(zernike_priors, expected_zernike_priors)


def test_get_zernike_prior_empty_data(model_params):
    empty_data = MockData(np.array([]), np.array([]), np.array([]), np.array([]))
    zernike_priors = get_zernike_prior(model_params, empty_data)
    assert zernike_priors.shape == tf.TensorShape([0])  # Check for empty array shape

def test_extract_star_data_valid_keys(mock_data):
    """Test extracting valid data from the dataset."""
    result = extract_star_data(mock_data, train_key="noisy_stars", test_key="stars")
    
    expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

def test_extract_star_data_masks(mock_data):
    """Test extracting star masks from the dataset."""
    result = extract_star_data(mock_data, train_key="masks", test_key="masks")
    
    expected = np.array([[1], [0], [0], [1]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

def test_extract_star_data_missing_key(mock_data):
    """Test that the function raises a KeyError when a key is missing."""
    with pytest.raises(KeyError, match="Missing keys in dataset: \\['invalid_key'\\]"):
        extract_star_data(mock_data, train_key="invalid_key", test_key="stars")

def test_extract_star_data_partially_missing_key(mock_data):
    """Test that the function raises a KeyError if only one key is missing."""
    with pytest.raises(KeyError, match="Missing keys in dataset: \\['missing_stars'\\]"):
        extract_star_data(mock_data, train_key="noisy_stars", test_key="missing_stars")


def test_extract_star_data_tensor_conversion(mock_data):
    """Test that the function properly converts TensorFlow tensors to NumPy arrays."""
    result = extract_star_data(mock_data, train_key="noisy_stars", test_key="stars")
    
    assert isinstance(result, np.ndarray), "The result should be a NumPy array"
    assert result.dtype == np.float32, "The NumPy array should have dtype float32"

def test_compute_centroid_correction_with_masks(mock_data):
    """Test compute_centroid_correction function with masks present."""
    # Given that compute_centroid_correction expects a model_params and data object
    model_params = RecursiveNamespace(
        pix_sampling=12e-6,  # Example pixel sampling in meters
        correct_centroids=True,
        reference_shifts=[-1/3, -1/3]
    )

    # Mock the internal function calls:
    with patch('wf_psf.data.training_preprocessing.extract_star_data') as mock_extract_star_data, \
         patch('wf_psf.data.training_preprocessing.compute_zernike_tip_tilt') as mock_compute_zernike_tip_tilt:
        
        # Mock the return values of extract_star_data and compute_zernike_tip_tilt
        mock_extract_star_data.side_effect = lambda data, train_key, test_key: (
            np.array([[1, 2], [3, 4]]) if train_key == 'noisy_stars' else np.array([[5, 6], [7, 8]])
        )
        mock_compute_zernike_tip_tilt.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Call the function under test
        result = compute_centroid_correction(model_params, mock_data)
        
        # Ensure the result has the correct shape
        assert result.shape == (2, 3)  # Should be (n_stars, 3 Zernike components)
        
        assert np.allclose(result[0, :], np.array([0, -0.1, -0.2]))  # First star Zernike coefficients
        assert np.allclose(result[1, :], np.array([0, -0.3, -0.4]))  # Second star Zernike coefficients


def test_compute_centroid_correction_without_masks(mock_data):
    """Test compute_centroid_correction function when no masks are provided."""
    # Remove masks from mock_data
    mock_data.test_data.dataset["masks"] = None
    mock_data.training_data.dataset["masks"] = None
    
    # Define model parameters
    model_params = RecursiveNamespace(
        pix_sampling=12e-6,  # Example pixel sampling in meters
        correct_centroids=True,
        reference_shifts=[-1/3, -1/3]
    )
    
    # Mock internal function calls
    with patch('wf_psf.data.training_preprocessing.extract_star_data') as mock_extract_star_data, \
         patch('wf_psf.data.training_preprocessing.compute_zernike_tip_tilt') as mock_compute_zernike_tip_tilt:
        
        # Mock extract_star_data to return synthetic star postage stamps
        mock_extract_star_data.side_effect = lambda data, train_key, test_key: (
            np.array([[1, 2], [3, 4]]) if train_key == 'noisy_stars' else np.array([[5, 6], [7, 8]])
        )
        
        # Mock compute_zernike_tip_tilt assuming no masks
        mock_compute_zernike_tip_tilt.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        # Call function under test
        result = compute_centroid_correction(model_params, mock_data)

        # Validate result shape
        assert result.shape == (2, 3)  # (n_stars, 3 Zernike components)

        # Validate expected values (adjust based on behavior)
        expected_result = np.array([
            [0, -0.1, -0.2],  # First star
            [0, -0.3, -0.4]   # Second star
        ])
        assert np.allclose(result, expected_result)
