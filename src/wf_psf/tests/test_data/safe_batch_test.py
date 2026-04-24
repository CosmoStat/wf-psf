import pytest
import numpy as np
from wf_psf.data.safe_batch import (
    _compute_valid_mask,
    safe_batch_builder,
    log_filtered_objects,
)


@pytest.fixture
def mock_dataset():
    """Fixture to create a mock .npy dataset for testing DataHandler."""
    # Mock dataset
    mock_dataset = {
        "positions": np.array([[1, 2], [3, 4]]),
        "noisy_stars": np.array([[5, 6], [7, 8]]),
        "stars": np.array([[5, 6], [7, 8]]),
        "SEDs": np.array(
            [
                [[400.0, 0.1], [500.0, 0.2], [600.0, 0.3]],  # Source 1
                [[400.0, 0.4], [500.0, 0.5], [600.0, 0.6]],  # Source 2
            ]
        ),
    }
    return mock_dataset


@pytest.fixture
def mock_NaN_dataset():
    """Fixture to create a mock .npy dataset for testing DataHandler."""
    # Mock dataset
    mock_dataset = {
        "positions": np.array([[1, 2], [np.nan, np.nan], [3, 4]]),
        "noisy_stars": np.array([[5, 6], [7, 8], [9, 10]]),
        "stars": np.array([[5, 6], [7, 8], [9, 10]]),
        "SEDs": np.array(
            [
                [[400.0, 0.1], [500.0, 0.2], [600.0, 0.3]],  # Source 1
                [[400.0, 0.4], [500.0, 0.5], [600.0, 0.6]],  # Source 2
                [[400.0, 0.7], [500.0, 0.8], [600.0, 0.9]],  # Source 3
            ]
        ),
    }
    return mock_dataset


def test_compute_valid_mask_all_valid(mock_dataset):
    anchor = mock_dataset["positions"]
    mask = _compute_valid_mask(anchor)

    assert mask.dtype == bool
    assert mask.shape == (2,)
    assert mask.all()


def test_compute_valid_mask_with_nan(mock_NaN_dataset):
    anchor = mock_NaN_dataset["positions"]
    mask = _compute_valid_mask(anchor)

    expected = np.array([True, False, True])
    np.testing.assert_array_equal(mask, expected)


def test_compute_valid_mask_all_nan():
    anchor = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    mask = _compute_valid_mask(anchor)

    expected = np.array([False, False])
    np.testing.assert_array_equal(mask, expected)


def test_compute_valid_mask_all_nan_1d():
    anchor = np.array([np.nan, np.nan, np.nan])

    mask = _compute_valid_mask(anchor)

    expected = np.array([False, False, False])
    np.testing.assert_array_equal(mask, expected)


def test_safe_batch_no_filtering(mock_dataset):
    anchor = mock_dataset["positions"]

    mask, filtered = safe_batch_builder(
        anchor,
        noisy_stars=mock_dataset["noisy_stars"],
        stars=mock_dataset["stars"],
        SEDs=mock_dataset["SEDs"],
    )

    assert mask.all()

    for key in ["noisy_stars", "stars", "SEDs"]:
        np.testing.assert_array_equal(filtered[key], mock_dataset[key])


def test_safe_batch_filters_correctly(mock_NaN_dataset):
    anchor = mock_NaN_dataset["positions"]

    mask, filtered = safe_batch_builder(
        anchor,
        noisy_stars=mock_NaN_dataset["noisy_stars"],
        stars=mock_NaN_dataset["stars"],
        SEDs=mock_NaN_dataset["SEDs"],
    )

    expected_mask = np.array([True, False, True])
    np.testing.assert_array_equal(mask, expected_mask)

    # Check alignment across all arrays
    np.testing.assert_array_equal(filtered["stars"], np.array([[5, 6], [9, 10]]))

    np.testing.assert_array_equal(filtered["noisy_stars"], np.array([[5, 6], [9, 10]]))

    assert filtered["SEDs"].shape[0] == 2


def test_safe_batch_all_nan_raises():
    anchor = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    stars = np.zeros((2, 2))

    with pytest.raises(ValueError, match="All samples"):
        safe_batch_builder(anchor, stars=stars)


def test_safe_batch_all_nan_returns_empty():
    anchor = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    stars = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="All samples were filtered out."):
        _, _ = safe_batch_builder(anchor, stars=stars)


def test_safe_batch_raises_on_invalid_type(mock_dataset):
    anchor = mock_dataset["positions"]

    with pytest.raises(TypeError):
        safe_batch_builder(
            anchor,
            stars="not an array",
        )


def test_safe_batch_filters_list_metadata(mock_NaN_dataset):
    anchor = mock_NaN_dataset["positions"]
    obj_ids = [101, 102, 103]

    _, filtered = safe_batch_builder(
        anchor,
        stars=mock_NaN_dataset["stars"],
        object_ids=obj_ids,
    )

    expected_ids = [101, 103]
    assert filtered["object_ids"] == expected_ids


def test_log_filtered_objects_all_removed():
    mask = np.array([False, False, False])
    obj_ids = np.array([10, 11, 12])

    class DummyLogger:
        def __init__(self):
            self.messages = []

        def warning(self, msg):
            self.messages.append(msg)

        def debug(self, msg):
            self.messages.append(msg)

    logger = DummyLogger()

    log_filtered_objects(mask, obj_ids, logger)

    assert any("3 samples removed" in msg for msg in logger.messages)
