
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import tensorflow as tf
from wf_psf.data.data_zernike_utils import (
    ZernikeInputs,
    ZernikeInputsFactory,
    get_np_zernike_prior,
    pad_contribution_to_order,
    combine_zernike_contributions,
    assemble_zernike_contributions,
    compute_zernike_tip_tilt,
)
from wf_psf.tests.test_data.test_data_utils import MockData, MockDataset
from types import SimpleNamespace as RecursiveNamespace


@pytest.fixture
def mock_model_params():
    return RecursiveNamespace(
        use_prior=True,
        correct_centroids=True,
        add_ccd_misalignments=True,
        param_hparams=RecursiveNamespace(n_zernikes=6),
    )

@pytest.fixture
def dummy_prior():
    return np.ones((4, 6), dtype=np.float32)

@pytest.fixture
def dummy_positions():
    return np.random.rand(4, 2).astype(np.float32)

@pytest.fixture
def dummy_centroid_dataset():
    return {"training": "dummy_train", "test": "dummy_test"}

def test_training_without_prior(mock_model_params):
    mock_model_params.use_prior = False
    data = MagicMock()
    data.training_dataset = {"positions": np.ones((2, 2))}
    data.test_dataset = {"positions": np.zeros((3, 2))}

    zinputs = ZernikeInputsFactory.build(data=data, run_type="training", model_params=mock_model_params)

    assert zinputs.centroid_dataset is data
    assert zinputs.zernike_prior is None
    np.testing.assert_array_equal(
        zinputs.misalignment_positions,
        np.concatenate([data.training_dataset["positions"], data.test_dataset["positions"]])
    )

@patch("wf_psf.data.data_zernike_utils.get_np_zernike_prior")
def test_training_with_dataset_prior(mock_get_prior, mock_model_params):
    mock_model_params.use_prior = True
    data = MagicMock()
    data.training_dataset = {"positions": np.ones((2, 2))}
    data.test_dataset = {"positions": np.zeros((2, 2))}
    mock_get_prior.return_value = np.array([1.0, 2.0, 3.0])

    zinputs = ZernikeInputsFactory.build(data=data, run_type="training", model_params=mock_model_params)

    assert zinputs.zernike_prior.tolist() == [1.0, 2.0, 3.0]
    mock_get_prior.assert_called_once_with(data)

def test_training_with_explicit_prior(mock_model_params, caplog):
    mock_model_params.use_prior = True
    data = MagicMock()
    data.training_dataset = {"positions": np.ones((1, 2))}
    data.test_dataset = {"positions": np.zeros((1, 2))}

    explicit_prior = np.array([9.0, 9.0, 9.0])

    with caplog.at_level("WARNING"):
        zinputs = ZernikeInputsFactory.build(data, "training", mock_model_params, prior=explicit_prior)

    assert "Zernike prior explicitly provided" in caplog.text
    assert (zinputs.zernike_prior == explicit_prior).all()


def test_inference_with_dict_and_prior(mock_model_params):
    mock_model_params.use_prior = True
    data = {
        "positions": np.ones((5, 2)),
        "zernike_prior": np.array([42.0, 0.0])
    }

    zinputs = ZernikeInputsFactory.build(data, "inference", mock_model_params)

    assert zinputs.centroid_dataset is None
    assert (zinputs.zernike_prior == data["zernike_prior"]).all()
    np.testing.assert_array_equal(zinputs.misalignment_positions, data["positions"])


def test_invalid_run_type(mock_model_params):
    data = {"positions": np.ones((2, 2))}
    with pytest.raises(ValueError, match="Unsupported run_type"):
        ZernikeInputsFactory.build(data, "invalid_mode", mock_model_params)


def test_get_np_zernike_prior():
    # Mock training and test data
    training_prior = np.array([[1, 2, 3], [4, 5, 6]])
    test_prior = np.array([[7, 8, 9]])

    # Construct fake DataConfigHandler structure using RecursiveNamespace
    data = RecursiveNamespace(
        training_data=RecursiveNamespace(dataset={"zernike_prior": training_prior}),
        test_data=RecursiveNamespace(dataset={"zernike_prior": test_prior})
    )

    expected_prior = np.concatenate((training_prior, test_prior), axis=0)

    result = get_np_zernike_prior(data)

    # Assert shape and values match expected
    np.testing.assert_array_equal(result, expected_prior)

def test_pad_contribution_to_order():
    # Input: batch of 2 samples, each with 3 Zernike coefficients
    input_contribution = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])
    
    max_order = 5  # Target size: pad to 5 coefficients

    expected_output = np.array([
        [1.0, 2.0, 3.0, 0.0, 0.0],
        [4.0, 5.0, 6.0, 0.0, 0.0],
    ])

    padded = pad_contribution_to_order(input_contribution, max_order)

    assert padded.shape == (2, 5), "Output shape should match padded shape"
    np.testing.assert_array_equal(padded, expected_output)


def test_no_padding_needed():
    """If current order equals max_order, return should be unchanged."""
    input_contribution = np.array([[1, 2, 3], [4, 5, 6]])
    max_order = 3
    output = pad_contribution_to_order(input_contribution, max_order)
    assert output.shape == input_contribution.shape
    np.testing.assert_array_equal(output, input_contribution)

def test_padding_to_much_higher_order():
    """Pad from order 2 to order 10."""
    input_contribution = np.array([[1, 2], [3, 4]])
    max_order = 10
    expected_output = np.hstack([input_contribution, np.zeros((2, 8))])
    output = pad_contribution_to_order(input_contribution, max_order)
    assert output.shape == (2, 10)
    np.testing.assert_array_equal(output, expected_output)

def test_empty_contribution():
    """Test behavior with empty input array (0 features)."""
    input_contribution = np.empty((3, 0))  # 3 samples, 0 coefficients
    max_order = 4
    expected_output = np.zeros((3, 4))
    output = pad_contribution_to_order(input_contribution, max_order)
    assert output.shape == (3, 4)
    np.testing.assert_array_equal(output, expected_output)


def test_zero_samples():
    """Test with zero samples (empty batch)."""
    input_contribution = np.empty((0, 3))  # 0 samples, 3 coefficients
    max_order = 5
    expected_output = np.empty((0, 5))
    output = pad_contribution_to_order(input_contribution, max_order)
    assert output.shape == (0, 5)
    np.testing.assert_array_equal(output, expected_output)


def test_combine_zernike_contributions_basic_case():
    """Combine two contributions with matching sample count and varying order."""
    contrib1 = np.array([[1, 2], [3, 4]])        # shape (2, 2)
    contrib2 = np.array([[5], [6]])              # shape (2, 1)
    expected = np.array([
        [1 + 5, 2 + 0],
        [3 + 6, 4 + 0]
    ])  # padded contrib2 to (2, 2)
    result = combine_zernike_contributions([contrib1, contrib2])
    np.testing.assert_array_equal(result, expected)

def test_combine_multiple_contributions():
    """Combine three contributions."""
    c1 = np.array([[1, 2, 3]])      # shape (1, 3)
    c2 = np.array([[4, 5]])         # shape (1, 2)
    c3 = np.array([[6]])            # shape (1, 1)
    expected = np.array([[1+4+6, 2+5+0, 3+0+0]])  # shape (1, 3)
    result = combine_zernike_contributions([c1, c2, c3])
    np.testing.assert_array_equal(result, expected)

def test_empty_input_list():
    """Raise ValueError when input list is empty."""
    with pytest.raises(ValueError, match="No contributions provided."):
        combine_zernike_contributions([])

def test_inconsistent_sample_count():
    """Raise error or produce incorrect shape if contributions have inconsistent sample counts."""
    c1 = np.array([[1, 2], [3, 4]])  # shape (2, 2)
    c2 = np.array([[5, 6]])          # shape (1, 2)
    with pytest.raises(ValueError):
        combine_zernike_contributions([c1, c2])

def test_single_contribution():
    """Combining a single contribution should return the same array (no-op)."""
    contrib = np.array([[7, 8, 9], [10, 11, 12]])
    result = combine_zernike_contributions([contrib])
    np.testing.assert_array_equal(result, contrib)

def test_zero_order_contributions():
    """Contributions with 0 Zernike coefficients."""
    contrib1 = np.empty((2, 0))  # 2 samples, 0 coefficients
    contrib2 = np.empty((2, 0))
    expected = np.empty((2, 0))
    result = combine_zernike_contributions([contrib1, contrib2])
    assert result.shape == (2, 0)
    np.testing.assert_array_equal(result, expected)

@patch("wf_psf.data.data_zernike_utils.compute_centroid_correction")
@patch("wf_psf.data.data_zernike_utils.compute_ccd_misalignment")
def test_full_contribution_combination(mock_ccd, mock_centroid, mock_model_params, dummy_prior, dummy_centroid_dataset, dummy_positions):
    mock_centroid.return_value = np.full((4, 6), 2.0)
    mock_ccd.return_value = np.full((4, 6), 3.0)

    result = assemble_zernike_contributions(
        model_params=mock_model_params,
        zernike_prior=dummy_prior,
        centroid_dataset=dummy_centroid_dataset,
        positions=dummy_positions
    )

    expected = dummy_prior + 2.0 + 3.0
    np.testing.assert_allclose(result.numpy(), expected)

def test_prior_only(mock_model_params, dummy_prior):
    mock_model_params.correct_centroids = False
    mock_model_params.add_ccd_misalignments = False

    result = assemble_zernike_contributions(
        model_params=mock_model_params,
        zernike_prior=dummy_prior,
        centroid_dataset=None,
        positions=None
    )

    np.testing.assert_array_equal(result.numpy(), dummy_prior)

def test_no_contributions_returns_zeros():
    model_params = RecursiveNamespace(
        use_prior=False,
        correct_centroids=False,
        add_ccd_misalignments=False,
        param_hparams=RecursiveNamespace(n_zernikes=8),
    )

    result = assemble_zernike_contributions(model_params)

    assert isinstance(result, tf.Tensor)
    assert result.shape == (1, 8)
    np.testing.assert_array_equal(result.numpy(), np.zeros((1, 8)))

def test_prior_as_tensor(mock_model_params):
    tensor_prior = tf.ones((4, 6), dtype=tf.float32)

    mock_model_params.correct_centroids = False
    mock_model_params.add_ccd_misalignments = False

    result = assemble_zernike_contributions(
        model_params=mock_model_params,
        zernike_prior=tensor_prior
    )

    assert isinstance(result, tf.Tensor)
    np.testing.assert_array_equal(result.numpy(), np.ones((4, 6)))

@patch("wf_psf.data.data_zernike_utils.compute_centroid_correction")
def test_inconsistent_shapes_raises_error(mock_centroid, mock_model_params, dummy_prior, dummy_centroid_dataset):
    mock_model_params.add_ccd_misalignments = False
    mock_centroid.return_value = np.ones((5, 6))  # 5 samples instead of 4

    with pytest.raises(ValueError, match="All contributions must have the same number of samples"):
        assemble_zernike_contributions(
            model_params=mock_model_params,
            zernike_prior=dummy_prior,
            centroid_dataset=dummy_centroid_dataset,
            positions=None
        )


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