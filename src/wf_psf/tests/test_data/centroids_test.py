"""UNIT TESTS FOR PACKAGE MODULE: CENTROIDS.

This module contains unit tests for the wf_psf.utils centroids module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import pytest
from wf_psf.data.centroids import compute_centroid_correction, CentroidEstimator
from wf_psf.data.data_handler import extract_star_data
from wf_psf.data.data_zernike_utils import compute_zernike_tip_tilt
from wf_psf.utils.read_config import RecursiveNamespace
from unittest.mock import MagicMock, patch


# Function to compute centroid based on first-order moments
def calculate_centroid(image, mask=None):
    if mask is not None:
        image = np.ma.masked_array(image, mask=mask)

    # Calculate moments
    M00 = np.sum(image)
    M10 = np.sum(np.arange(image.shape[1]) * np.sum(image, axis=0))
    M01 = np.sum(np.arange(image.shape[0]) * np.sum(image, axis=1))

    # Centroid formula
    xc = M10 / M00
    yc = M01 / M00
    return (xc, yc)


@pytest.fixture
def simple_star_and_mask():
    """Fixture for an image with multiple non-zero pixels for centroid calculation."""
    num_images = 1  # Change this to test with multiple images
    image = np.zeros(
        (num_images, 5, 5)
    )  # Create a 3D array (5x5 image for each "image")

    # Place non-zero values in multiple pixels
    image[:, 2, 2] = 10  # Star at the center
    image[:, 2, 3] = 10  # Adjacent pixel
    image[:, 3, 2] = 10  # Adjacent pixel
    image[:, 3, 3] = 10  # Adjacent pixel forming a symmetric pattern

    mask = np.zeros_like(image)
    mask[:, 3, 2] = 1
    mask[:, 3, 3] = 1

    return image, mask


@pytest.fixture
def simple_image_with_mask(simple_image):
    """Fixture for a batch of star images with masks."""
    num_images = simple_image.shape[
        0
    ]  # Get the number of images from the first dimension
    mask = np.ones((num_images, 5, 5))  # Create a batch of masks
    mask[:, 1:4, 1:4] = 0  # Mask a 3x3 region for each image
    return simple_image, mask


@pytest.fixture
def centroid_estimator(simple_image):
    """Fixture for initializing CentroidEstimator."""
    return CentroidEstimator(simple_image)


@pytest.fixture
def centroid_estimator_with_mask(simple_image_with_mask):
    """Fixture for initializing CentroidEstimator with a mask."""
    image, mask = simple_image_with_mask
    return CentroidEstimator(image, mask=mask)


@pytest.fixture
def simple_image_with_centroid(simple_image):
    """Fixture for a simple image with known centroid and initial position."""
    image = simple_image

    # Known centroid and initial position (xc0, yc0) - for testing
    xc0, yc0 = 2.0, 2.0  # Assume the initial center of the image is (2.0, 2.0)

    # Create CentroidEstimator instance
    centroid_estimator = CentroidEstimator(im=image, n_iter=1)

    centroid_estimator.window = np.ones_like(image)
    centroid_estimator.xc0 = xc0
    centroid_estimator.yc0 = yc0

    # Simulate the computed centroid being slightly off-center
    centroid_estimator.xc = 2.3
    centroid_estimator.yc = 2.7

    return centroid_estimator


@pytest.fixture
def batch_images():
    """Fixture for multiple PSF images."""
    num_images = 3
    images = np.zeros((num_images, 5, 5))
    images[:, 2, 2] = 1  # Central pixel is the same for all images
    return images


def test_compute_centroid_correction_with_masks(mock_data):
    """Test compute_centroid_correction function with masks present."""
    # Given that compute_centroid_correction expects a model_params and data object
    model_params = RecursiveNamespace(
        pix_sampling=12e-6,  # Example pixel sampling in meters
        correct_centroids=True,
        reference_shifts=["-1/3", "-1/3"],
    )

    # Mock the internal function calls:
    with (
        patch("wf_psf.data.centroids.extract_star_data") as mock_extract_star_data,
        patch(
            "wf_psf.data.centroids.compute_zernike_tip_tilt"
        ) as mock_compute_zernike_tip_tilt,
    ):

        # Mock the return values of extract_star_data and compute_zernike_tip_tilt
        mock_extract_star_data.side_effect = lambda data, train_key, test_key: (
            np.array([[1, 2], [3, 4]])
            if train_key == "noisy_stars"
            else np.array([[5, 6], [7, 8]])
        )
        mock_compute_zernike_tip_tilt.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Call the function under test
        result = compute_centroid_correction(model_params, mock_data)

        # Ensure the result has the correct shape
        assert result.shape == (4, 3)  # Should be (n_stars, 3 Zernike components)

        assert np.allclose(
            result[0, :], np.array([0, -0.1, -0.2])
        )  # First star Zernike coefficients
        assert np.allclose(
            result[1, :], np.array([0, -0.3, -0.4])
        )  # Second star Zernike coefficients


def test_compute_centroid_correction_without_masks(mock_data):
    """Test compute_centroid_correction function when no masks are provided."""
    # Remove masks from mock_data
    mock_data.test_data.dataset["masks"] = None
    mock_data.training_data.dataset["masks"] = None

    # Define model parameters
    model_params = RecursiveNamespace(
        pix_sampling=12e-6,  # Example pixel sampling in meters
        correct_centroids=True,
        reference_shifts=["-1/3", "-1/3"],
    )

    # Mock internal function calls
    with (
        patch("wf_psf.data.centroids.extract_star_data") as mock_extract_star_data,
        patch(
            "wf_psf.data.centroids.compute_zernike_tip_tilt"
        ) as mock_compute_zernike_tip_tilt,
    ):

        # Mock extract_star_data to return synthetic star postage stamps
        mock_extract_star_data.side_effect = lambda data, train_key, test_key: (
            np.array([[1, 2], [3, 4]])
            if train_key == "noisy_stars"
            else np.array([[5, 6], [7, 8]])
        )

        # Mock compute_zernike_tip_tilt assuming no masks
        mock_compute_zernike_tip_tilt.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        # Call function under test
        result = compute_centroid_correction(model_params, mock_data)

        # Validate result shape
        assert result.shape == (4, 3)  # (n_stars, 3 Zernike components)

        # Validate expected values (adjust based on behavior)
        expected_result = np.array(
            [
                [0, -0.1, -0.2],  # From training data
                [0, -0.3, -0.4],
                [0, -0.1, -0.2],  # From test data (reused mocked return)
                [0, -0.3, -0.4],
            ]
        )
        assert np.allclose(result, expected_result)


# Test for centroid calculation without mask
def test_centroid_calculation_one_star(centroid_estimator):
    """Test that the centroid is calculated correctly for a simple image."""
    xc, yc = centroid_estimator.estimate()
    # The centroid should be at the center of the image
    assert np.isclose(xc, 2.0)
    assert np.isclose(yc, 2.0)


# Test for centroid calculation with mask
def test_centroid_calculation_with_one_star_and_mask(centroid_estimator_with_mask):
    """Test that the centroid is calculated correctly when a mask is applied."""
    xc, yc = centroid_estimator_with_mask.estimate()
    # The centroid should be shifted because the mask excludes part of the image
    assert np.isclose(xc, 2.0)
    assert np.isclose(yc, 2.0)


def test_centroid_calculation_multiple_images(multiple_images):
    """Test the centroid estimation for a batch of images."""
    estimator = CentroidEstimator(im=multiple_images)

    # Check that centroids are correctly estimated
    expected_centroids = [(2.0, 2.0), (1.0, 3.0), (3.0, 1.0)]
    for i, (xc, yc) in enumerate(zip(estimator.xc, estimator.yc)):
        assert np.isclose(xc, expected_centroids[i][0])
        assert np.isclose(yc, expected_centroids[i][1])


def test_centroid_no_mask(simple_star_and_mask):
    # Extract star
    star, _ = simple_star_and_mask

    # Expected centroid for the symmetric pattern
    true_centroid = (2.5, 2.5)

    # Create the CentroidEstimator instance (assuming auto_run=True by default)
    centroid_estimator = CentroidEstimator(im=star, n_iter=1)

    # Check if the centroid is calculated correctly
    computed_centroid = (centroid_estimator.xc, centroid_estimator.yc)
    assert np.isclose(computed_centroid[0], true_centroid[0])
    assert np.isclose(computed_centroid[1], true_centroid[1])


# Test for centroid calculation with a mask
def test_centroid_with_mask(simple_star_and_mask):
    # Extract star and mask
    star, mask = simple_star_and_mask

    # Expected centroid after masking (estimated manually)
    expected_masked_centroid = (2.0, 2.5)

    # Create the CentroidEstimator instance (with mask)
    centroid_estimator = CentroidEstimator(im=star, mask=mask, n_iter=1)

    # Check if the centroid is calculated correctly with the mask applied
    computed_centroid = (centroid_estimator.xc, centroid_estimator.yc)
    assert np.isclose(computed_centroid[0], expected_masked_centroid[0])
    assert np.isclose(computed_centroid[1], expected_masked_centroid[1])


def test_centroid_estimator_initialization(simple_image):
    """Test the initialization of the CentroidEstimator."""
    estimator = CentroidEstimator(simple_image)
    assert estimator.im.shape == (1, 5, 5)  # Shape should match the input image
    assert (
        estimator.xc0 == 2.5
    )  # Default xc should be the center of the image, i.e. float(self.stamp_size[0]) / 2
    assert (
        estimator.yc0 == 2.5
    )  # Default yc should be the center of the image, i.e. float(self.stamp_size[0]) / 2
    assert estimator.sigma_init == 7.5  # Default sigma_init should be 7.5
    assert estimator.n_iter == 5  # Default number of iterations should be 5
    assert estimator.mask is None  # By default, mask should be None


def test_single_iteration(centroid_estimator):
    """Test that the internal methods are called exactly once for n_iter=1."""
    # Mock the methods
    centroid_estimator.update_grid = MagicMock()
    centroid_estimator.elliptical_gaussian = MagicMock()
    centroid_estimator.compute_moments = MagicMock()

    # Set n_iter to 1
    centroid_estimator.n_iter = 1

    # Run the estimate method
    centroid_estimator.estimate()

    # Assert that the methods were called only once
    centroid_estimator.update_grid.assert_called_once()
    centroid_estimator.elliptical_gaussian.assert_called_once()
    centroid_estimator.compute_moments.assert_called_once()


def test_single_iteration_auto_run(simple_image):
    """Test that the internal methods are called exactly once for n_iter=1."""
    # Patch the methods at the time the object is created
    with (
        patch.object(CentroidEstimator, "update_grid") as update_grid_mock,
        patch.object(
            CentroidEstimator, "elliptical_gaussian"
        ) as elliptical_gaussian_mock,
        patch.object(CentroidEstimator, "compute_moments") as compute_moments_mock,
    ):

        # Initialize the CentroidEstimator with auto_run=True
        _ = CentroidEstimator(im=simple_image, n_iter=1, auto_run=True)

        # Assert that the methods were called only once
        update_grid_mock.assert_called_once()
        elliptical_gaussian_mock.assert_called_once()
        compute_moments_mock.assert_called_once()


def test_update_grid(simple_image):
    """Test that the grid is correctly updated."""
    centroid_estimator = CentroidEstimator(im=simple_image, auto_run=True, n_iter=1)

    # Check the shapes of the grid coordinates
    assert centroid_estimator.xx.shape == (1, 5, 5)
    assert centroid_estimator.yy.shape == (1, 5, 5)

    # Check the values of the grid coordinates
    # xx should be the same for all rows and columns (broadcasted across the image)
    assert np.allclose(
        centroid_estimator.xx,
        np.array(
            [
                [
                    [
                        [-2.5, -2.5, -2.5, -2.5, -2.5],
                        [-1.5, -1.5, -1.5, -1.5, -1.5],
                        [-0.5, -0.5, -0.5, -0.5, -0.5],
                        [0.5, 0.5, 0.5, 0.5, 0.5],
                        [1.5, 1.5, 1.5, 1.5, 1.5],
                    ]
                ]
            ]
        ),
    )

    # yy should be the same for all columns and rows (broadcasted across the image)
    assert np.allclose(
        centroid_estimator.yy,
        np.array(
            [
                [
                    [
                        [-2.5, -1.5, -0.5, 0.5, 1.5],
                        [-2.5, -1.5, -0.5, 0.5, 1.5],
                        [-2.5, -1.5, -0.5, 0.5, 1.5],
                        [-2.5, -1.5, -0.5, 0.5, 1.5],
                        [-2.5, -1.5, -0.5, 0.5, 1.5],
                    ]
                ]
            ]
        ),
    )


def test_elliptical_gaussian(simple_image):
    """Test that the elliptical Gaussian is calculated correctly."""
    centroid_estimator = CentroidEstimator(im=simple_image, n_iter=1)
    # Check if the output is a valid 2D array with the correct shape
    assert centroid_estimator.window.shape == (1, 5, 5)

    # Check if the Gaussian window values are reasonable (non-negative and decrease with distance)
    assert np.all(centroid_estimator.window >= 0)
    assert np.isclose(np.sum(centroid_estimator.window), 25, atol=1.0)


def test_intra_pixel_shifts(simple_image_with_centroid):
    """Test the return_intra_pixel_shifts method."""
    centroid_estimator = simple_image_with_centroid

    # Calculate intra-pixel shifts
    shifts = centroid_estimator.get_intra_pixel_shifts()

    # Expected intra-pixel shifts
    expected_x_shift = 2.3 - 2.0  # xc - xc0
    expected_y_shift = 2.7 - 2.0  # yc - yc0

    # Check that the shifts are correct
    assert np.isclose(
        shifts[0], expected_x_shift
    ), f"Expected {expected_x_shift}, got {shifts[0]}"
    assert np.isclose(
        shifts[1], expected_y_shift
    ), f"Expected {expected_y_shift}, got {shifts[1]}"
