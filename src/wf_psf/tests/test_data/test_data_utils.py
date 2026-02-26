"""Helper classes and functions for testing TensorFlow data conversion."""

import tensorflow as tf


class MockDataset:
    """MockDataset for testing TensorFlow conversion.

    Parameters
    ----------
    positions : tf.Tensor
        TensorFlow-formatted positions with shape [batch_size, 2].
    zernike_priors : tf.Tensor
            TensorFlow-formatted zernike priors with shape [batch_size, n_zernikes].
    star_type : str
        The key to use for star data in the dataset (e.g., 'noisy_stars' or 'stars').
    stars : tf.Tensor
        TensorFlow-formatted star images with shape [batch_size, H, W].
    masks : tf.Tensor
        TensorFlow-formatted masks with shape [batch_size, H, W].
    """

    def __init__(self, positions, zernike_priors, star_type, stars, masks):
        self.dataset = {
            "positions": positions,
            "zernike_prior": zernike_priors,
            star_type: stars,
            "masks": masks,
        }


class MockData:
    """
    MockData for testing TensorFlow conversion of both training and test datasets.

    Parameters
    ----------
    training_positions : tf.Tensor
        TensorFlow-formatted training positions with shape [batch_size, 2].
    test_positions : tf.Tensor
        TensorFlow-formatted test positions with shape [batch_size, 2].
    training_zernike_priors : tf.Tensor
        TensorFlow-formatted training zernike priors with shape [batch_size, n_zernikes].
    test_zernike_priors : tf.Tensor
        TensorFlow-formatted test zernike priors with shape [batch_size, n_zernikes].
    noisy_stars : tf.Tensor
        TensorFlow-formatted noisy star images for training with shape [batch_size, H, W].
    noisy_masks : tf.Tensor
        TensorFlow-formatted masks for training with shape [batch_size, H, W].
    stars : tf.Tensor
        TensorFlow-formatted star images for testing with shape [batch_size, H, W].
    masks : tf.Tensor
        TensorFlow-formatted masks for testing with shape [batch_size, H, W].
    """

    def __init__(
        self,
        training_positions,
        test_positions,
        training_zernike_priors=None,
        test_zernike_priors=None,
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
            masks=noisy_masks,
        )
        self.test_data = MockDataset(
            positions=test_positions,
            zernike_priors=test_zernike_priors,
            star_type="stars",
            stars=stars,
            masks=masks,
        )


def assert_tensor(value, expected_shape=None, expected_dtype=tf.float32):
    """Assert value is a TensorFlow tensor with expected shape and dtype."""
    assert tf.is_tensor(value), f"Expected tf.Tensor, got {type(value)}"
    assert value.dtype == expected_dtype, (
        f"Expected dtype {expected_dtype}, got {value.dtype}"
    )
    if expected_shape is not None:
        assert tuple(value.shape) == expected_shape, (
            f"Expected shape {expected_shape}, got {tuple(value.shape)}"
        )
