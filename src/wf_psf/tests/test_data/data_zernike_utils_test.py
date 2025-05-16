
import pytest
import numpy as np
import tensorflow as tf
from wf_psf.data.data_zernike_utils import (
    get_zernike_prior,
    compute_zernike_tip_tilt,
)
from wf_psf.tests.test_data.test_data_utils import MockData, MockDataset

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


def test_compute_zernike_tip_tilt_single_batch(mocker, simple_image, identity_mask):
    """Test compute_zernike_tip_tilt with single batch input and mocks."""

    # Mock the CentroidEstimator class
    mock_centroid_calc = mocker.patch("wf_psf.data.centroids.CentroidEstimator", autospec=True)

    # Create a mock instance and configure get_intra_pixel_shifts()
    mock_instance = mock_centroid_calc.return_value
    mock_instance.get_intra_pixel_shifts.return_value = np.array([[0.05, -0.02]])  # Shape (1, 2)

    # Mock shift_x_y_to_zk1_2_wavediff to return predictable values
    mock_shift_fn = mocker.patch(
        "wf_psf.data.data_zernike_utils.shift_x_y_to_zk1_2_wavediff",
        side_effect=lambda shift: shift * 0.5  # Mocked conversion for test
    )

    # Define test inputs (batch of 1 image)
    pixel_sampling = 12e-6
    reference_shifts = [-1 / 3, -1 / 3]  # Default Euclid conditions

    # Run the function
    zernike_corrections = compute_zernike_tip_tilt(simple_image, identity_mask, pixel_sampling, reference_shifts)
    zernike_corrections = compute_zernike_tip_tilt(simple_image, identity_mask, pixel_sampling, reference_shifts)

    # Expected shifts based on centroid calculation
    expected_dx = (reference_shifts[1] - (-0.02)) # Expected x-axis shift in meters
    expected_dy = (reference_shifts[0] - 0.05) # Expected y-axis shift in meters

    # Expected calls to the mocked function
    # Extract the arguments passed to mock_shift_fn
    args, _ = mock_shift_fn.call_args_list[0]  # Get the first call args 

    # Compare expected values with the actual arguments passed to the mock function
    np.testing.assert_allclose(args[0][0], expected_dx * pixel_sampling, rtol=1e-7, atol=0)

    # Check dy values similarly
    np.testing.assert_allclose(args[0][1], expected_dy * pixel_sampling, rtol=1e-7, atol=0)

    # Expected values based on mock side_effect (0.5 * shift)
    np.testing.assert_allclose(zernike_corrections[0, 0], expected_dx * pixel_sampling * 0.5)  # Zk1
    np.testing.assert_allclose(zernike_corrections[0, 1], expected_dy * pixel_sampling * 0.5) # Zk2

def test_compute_zernike_tip_tilt_batch(mocker, multiple_images):
    """Test compute_zernike_tip_tilt with batch input and mocks."""
    
    # Mock the CentroidEstimator class
    mock_centroid_calc = mocker.patch("wf_psf.data.centroids.CentroidEstimator", autospec=True)

    # Create a mock instance and configure get_intra_pixel_shifts()
    mock_instance = mock_centroid_calc.return_value
    mock_instance.get_intra_pixel_shifts.return_value = np.array([[0.05, -0.02], [0.04, -0.01], [0.06, -0.03]])  # Shape (3, 2)

    # Mock shift_x_y_to_zk1_2_wavediff to return predictable values
    mock_shift_fn = mocker.patch(
        "wf_psf.data.data_zernike_utils.shift_x_y_to_zk1_2_wavediff",
        side_effect=lambda shift: shift * 0.5  # Mocked conversion for test
    )

    # Define test inputs (batch of 3 images)
    pixel_sampling = 12e-6
    reference_shifts = [-1 / 3, -1 / 3]  # Default Euclid conditions

    # Run the function
    zernike_corrections = compute_zernike_tip_tilt(
        star_images=multiple_images, 
        pixel_sampling=pixel_sampling, 
        reference_shifts=reference_shifts
        )

    # Check if the mock function was called once with the full batch
    assert len(mock_shift_fn.call_args_list) == 1, f"Expected 1 call, but got {len(mock_shift_fn.call_args_list)}"

    # Get the arguments passed to the mock function for the batch of images
    args, _ = mock_shift_fn.call_args_list[0]  

    print("Shape of args[0]:", args[0].shape)
    print("Contents of args[0]:", args[0])
    print("Mock function call args list:", mock_shift_fn.call_args_list)

    # Reshape args[0] to (N, 2) for batch processing
    args_array = np.array(args[0]).reshape(-1, 2)

    # Process the displacements and expected values for each image in the batch
    expected_dx = reference_shifts[1] - mock_instance.get_intra_pixel_shifts.return_value[:, 1]  # Expected x-axis shift in meters
   
    expected_dy = reference_shifts[0] - mock_instance.get_intra_pixel_shifts.return_value[:, 0]  # Expected y-axis shift in meters

    # Compare expected values with the actual arguments passed to the mock function
    np.testing.assert_allclose(args_array[:, 0], expected_dx * pixel_sampling, rtol=1e-7, atol=0)
    np.testing.assert_allclose(args_array[:, 1], expected_dy * pixel_sampling, rtol=1e-7, atol=0)

    # Expected values based on mock side_effect (0.5 * shift)
    np.testing.assert_allclose(zernike_corrections[:, 0], expected_dx * pixel_sampling * 0.5)  # Zk1 for each image
    np.testing.assert_allclose(zernike_corrections[:, 1], expected_dy * pixel_sampling * 0.5)  # Zk2 for each image