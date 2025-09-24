import pytest
import numpy as np
import tensorflow as tf
from wf_psf.data.data_handler import (
    DataHandler,
    get_data_array,
    extract_star_data,
)
from wf_psf.utils.read_config import RecursiveNamespace


def mock_sed():
    # Create a fake SED with shape (n_wavelengths,) — match what your real SEDs look like
    return np.linspace(0.1, 1.0, 50)


def mock_sed():
    # Create a fake SED with shape (n_wavelengths,) — match what your real SEDs look like
    return np.linspace(0.1, 1.0, 50)


def test_process_sed_data_auto_load(data_params, simPSF):
    # load_data=True → dataset is used and SEDs processed automatically
    data_handler = DataHandler(
        "training", data_params.training, simPSF, n_bins_lambda=10, load_data=True
    )
    assert data_handler.sed_data is not None
    assert data_handler.sed_data.shape[1] == 10  # n_bins_lambda


def test_load_train_dataset(tmp_path, simPSF):
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
    data_params = RecursiveNamespace(data_dir=str(data_dir), file="train_data.npy")

    n_bins_lambda = 10
    data_handler = DataHandler(
        "training", data_params, simPSF, n_bins_lambda, load_data=False
    )

    # Call the load_dataset method
    data_handler.load_dataset()

    # Assertions
    assert np.array_equal(data_handler.dataset["positions"], mock_dataset["positions"])
    assert np.array_equal(
        data_handler.dataset["noisy_stars"], mock_dataset["noisy_stars"]
    )
    assert np.array_equal(data_handler.dataset["SEDs"], mock_dataset["SEDs"])


def test_load_test_dataset(tmp_path, simPSF):
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
    data_params = RecursiveNamespace(data_dir=str(data_dir), file="test_data.npy")

    n_bins_lambda = 10
    data_handler = DataHandler(
        dataset_type="test",
        data_params=data_params,
        simPSF=simPSF,
        n_bins_lambda=n_bins_lambda,
        load_data=False,
    )

    # Call the load_dataset method
    data_handler.load_dataset()

    # Assertions
    assert np.array_equal(data_handler.dataset["positions"], mock_dataset["positions"])
    assert np.array_equal(data_handler.dataset["stars"], mock_dataset["stars"])
    assert np.array_equal(data_handler.dataset["SEDs"], mock_dataset["SEDs"])


def test_validate_train_dataset_missing_noisy_stars_raises(tmp_path, simPSF):
    """Test that validation raises an error if 'noisy_stars' is missing in training data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    temp_data_file = data_dir / "train_data.npy"

    mock_dataset = {
        "positions": np.array([[1, 2], [3, 4]]),  # No 'noisy_stars' key
        "SEDs": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
    }

    np.save(temp_data_file, mock_dataset)

    data_params = RecursiveNamespace(data_dir=str(data_dir), file="train_data.npy")

    n_bins_lambda = 10
    data_handler = DataHandler(
        "training", data_params, simPSF, n_bins_lambda, load_data=False
    )

    with pytest.raises(
        ValueError, match="Missing required field 'noisy_stars' in training dataset."
    ):
        data_handler.load_dataset()
        data_handler.validate_and_process_dataset()


def test_load_test_dataset_missing_stars(tmp_path, simPSF):
    """Test that a warning is raised if 'stars' is missing in test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    temp_data_file = data_dir / "test_data.npy"

    mock_dataset = {
        "positions": np.array([[1, 2], [3, 4]]),  # No 'stars' key
        "SEDs": np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
    }

    np.save(temp_data_file, mock_dataset)

    data_params = RecursiveNamespace(data_dir=str(data_dir), file="test_data.npy")

    n_bins_lambda = 10
    data_handler = DataHandler(
        "test", data_params, simPSF, n_bins_lambda, load_data=False
    )

    with pytest.raises(
        ValueError, match="Missing required field 'stars' in test dataset."
    ):
        data_handler.load_dataset()
        data_handler.validate_and_process_dataset()


def test_extract_star_data_valid_keys(mock_data):
    """Test extracting valid data from the dataset."""
    result = extract_star_data(mock_data, train_key="noisy_stars", test_key="stars")

    expected = tf.concat(
        [
            tf.constant(
                [np.arange(25).reshape(5, 5), np.arange(25, 50).reshape(5, 5)],
                dtype=tf.float32,
            ),
            tf.constant([np.full((5, 5), 100), np.full((5, 5), 200)], dtype=tf.float32),
        ],
        axis=0,
    )

    np.testing.assert_array_equal(result, expected)


def test_extract_star_data_masks(mock_data):
    """Test extracting star masks from the dataset."""
    result = extract_star_data(mock_data, train_key="masks", test_key="masks")

    mask0 = np.eye(5, dtype=np.float32)
    mask1 = np.ones((5, 5), dtype=np.float32)
    mask2 = np.zeros((5, 5), dtype=np.float32)
    mask3 = np.tri(5, dtype=np.float32)

    expected = np.array([mask0, mask1, mask2, mask3], dtype=np.float32)

    np.testing.assert_array_equal(result, expected)


def test_extract_star_data_missing_key(mock_data):
    """Test that the function raises a KeyError when a key is missing."""
    with pytest.raises(KeyError, match="Missing keys in dataset: \\['invalid_key'\\]"):
        extract_star_data(mock_data, train_key="invalid_key", test_key="stars")


def test_extract_star_data_partially_missing_key(mock_data):
    """Test that the function raises a KeyError if only one key is missing."""
    with pytest.raises(
        KeyError, match="Missing keys in dataset: \\['missing_stars'\\]"
    ):
        extract_star_data(mock_data, train_key="noisy_stars", test_key="missing_stars")


def test_extract_star_data_tensor_conversion(mock_data):
    """Test that the function properly converts TensorFlow tensors to NumPy arrays."""
    result = extract_star_data(mock_data, train_key="noisy_stars", test_key="stars")
    assert isinstance(result, np.ndarray), "The result should be a NumPy array"
    assert result.dtype == np.float32, "The NumPy array should have dtype float32"


def test_reference_shifts_broadcasting():
    reference_shifts = [-1 / 3, -1 / 3]  # Example reference_shifts
    shifts = np.random.rand(2, 2400)  # Example shifts array

    # Ensure reference_shifts is a NumPy array (if it's not already)
    reference_shifts = np.array(reference_shifts)

    # Broadcast reference_shifts to match the shape of shifts
    reference_shifts = np.broadcast_to(
        reference_shifts[:, None], shifts.shape
    )  # Shape will be (2, 2400)

    # Ensure shapes are compatible for subtraction
    displacements = reference_shifts - shifts

    # Test the result
    assert displacements.shape == shifts.shape, "Shapes do not match"
    assert np.all(displacements.shape == (2, 2400)), "Broadcasting failed"


@pytest.mark.parametrize(
    "run_type,data_fixture,key,train_key,test_key,allow_missing,expect",
    [
        # ===================================================
        # training/simulation/metrics → extract_star_data path
        # ===================================================
        (
            "training",
            "mock_data",
            None,
            "positions",
            None,
            False,
            np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        ),
        (
            "simulation",
            "mock_data",
            "none",
            "noisy_stars",
            "stars",
            True,
            # will concatenate noisy_stars from train and stars from test
            # expected shape: (4, 5, 5)
            # validate shape only, not full content (too large)
            "shape:(4, 5, 5)",
        ),
        (
            "metrics",
            "mock_data",
            "zernike_prior",
            None,
            None,
            True,
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
        ),
        # =================
        # inference (success)
        # =================
        (
            "inference",
            "mock_data_inference",
            "positions",
            None,
            None,
            False,
            np.array([[9, 9], [10, 10]]),
        ),
        (
            "inference",
            "mock_data_inference",
            "zernike_prior",
            None,
            None,
            False,
            np.array([[0.9, 0.9]]),
        ),
        # ==============================
        # inference → allow_missing=True
        # ==============================
        (
            "inference",
            "mock_data_inference",
            None,
            None,
            None,
            True,
            None,
        ),
        (
            "inference",
            "mock_data_inference",
            "missing_key",
            None,
            None,
            True,
            None,
        ),
        # =================================
        # inference → allow_missing=False → errors
        # =================================
        (
            "inference",
            "mock_data_inference",
            None,
            None,
            None,
            False,
            pytest.raises(ValueError),
        ),
        (
            "inference",
            "mock_data_inference",
            "missing_key",
            None,
            None,
            False,
            pytest.raises(KeyError),
        ),
    ],
)
def test_get_data_array_v2(
    request, run_type, data_fixture, key, train_key, test_key, allow_missing, expect
):
    data = request.getfixturevalue(data_fixture)

    if hasattr(expect, "__enter__") and hasattr(expect, "__exit__"):
        with expect:
            get_data_array(
                data,
                run_type,
                key=key,
                train_key=train_key,
                test_key=test_key,
                allow_missing=allow_missing,
            )
        return

    result = get_data_array(
        data,
        run_type,
        key=key,
        train_key=train_key,
        test_key=test_key,
        allow_missing=allow_missing,
    )

    if expect is None:
        assert result is None
    elif isinstance(expect, str) and expect.startswith("shape:"):
        expected_shape = tuple(eval(expect.replace("shape:", "")))
        assert isinstance(result, np.ndarray)
        assert result.shape == expected_shape
    else:
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, expect, rtol=1e-6, atol=1e-8)
