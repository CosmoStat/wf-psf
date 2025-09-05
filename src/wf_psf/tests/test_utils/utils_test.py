"""UNIT TESTS FOR PACKAGE MODULE: UTILS.

This module contains unit tests for the wf_psf.utils utils module.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import tensorflow as tf
import numpy as np
from wf_psf.utils.utils import (
    NoiseEstimator,
    zernike_generator,
    compute_unobscured_zernike_projection,
    decompose_tf_obscured_opd_basis,
)
from wf_psf.sims.psf_simulator import PSFSimulator


def test_initialization():
    """Test if NoiseEstimator initializes correctly."""
    img_dim = (50, 50)
    win_rad = 10
    estimator = NoiseEstimator(img_dim, win_rad)

    assert estimator.img_dim == img_dim
    assert estimator.win_rad == win_rad
    assert isinstance(estimator.window, np.ndarray)
    assert estimator.window.shape == img_dim


def test_init_window():
    """Test that the exclusion window is correctly initialized."""
    img_dim = (50, 50)
    win_rad = 10
    estimator = NoiseEstimator(img_dim, win_rad)

    mid_x, mid_y = img_dim[0] / 2, img_dim[1] / 2

    for x in range(img_dim[0]):
        for y in range(img_dim[1]):
            # Pixel inside the exclusion radius should be False, others True
            inside_radius = np.sqrt((x - mid_x) ** 2 + (y - mid_y) ** 2) <= win_rad
            assert estimator.window[x, y] == (not inside_radius)


def test_sigma_mad():
    """Test the MAD-based standard deviation estimation."""
    data = np.array(
        [1, 1, 2, 2, 3, 3, 4, 4, 100]
    )  # Outlier should not heavily influence MAD
    expected_sigma = 1.4826 * np.median(np.abs(data - np.median(data)))

    assert np.isclose(NoiseEstimator.sigma_mad(data), expected_sigma, atol=1e-4)


def test_estimate_noise_without_default_window():
    """Test noise estimation with the default exclusion window (no custom mask)."""
    img_dim = (50, 50)
    win_rad = 5
    estimator = NoiseEstimator(img_dim, win_rad)

    # Create a synthetic noisy image (Gaussian noise with mean=0, std=10)
    np.random.seed(42)
    image = np.random.normal(0, 10, img_dim)

    noise_estimation = estimator.estimate_noise(image)

    # The estimated noise should be close to 10 (the true std)
    assert np.isclose(noise_estimation, 10, atol=2)


def test_estimate_noise_with_custom_mask():
    """Test noise estimation with a custom mask applied outside the exclusion radius."""
    img_dim = (50, 50)
    estimator = NoiseEstimator(img_dim, win_rad=5)

    # Create synthetic noise with std=5
    np.random.seed(42)
    image = np.random.normal(0, 5, img_dim)

    # Custom mask excluding top-left corner
    custom_mask = np.ones(img_dim, dtype=bool)
    custom_mask[:10, :10] = False  # Mask out top-left 10x10 pixels

    noise_estimation = estimator.estimate_noise(image, mask=custom_mask)

    assert np.isclose(noise_estimation, 5, atol=1)


def test_apply_mask_with_none_mask():
    """Test apply_mask when mask is None."""
    img_dim = (10, 10)
    estimator = NoiseEstimator(img_dim, win_rad=3)

    result = estimator.apply_mask(None)  # Pass None as the mask

    # It should return the window itself when no mask is provided
    assert np.array_equal(
        result, estimator.window
    ), "apply_mask should return the window when mask is None."


def test_apply_mask_with_valid_mask():
    """Test apply_mask when a valid mask is provided."""
    img_dim = (10, 10)
    estimator = NoiseEstimator(img_dim, win_rad=3)

    # Create a custom mask
    custom_mask = np.ones(img_dim, dtype=bool)
    custom_mask[5, 5] = False  # Set a pixel to False to exclude it from the window

    result = estimator.apply_mask(custom_mask)

    # Check that the mask was applied correctly: pixel (5, 5) should be False, others True
    expected_result = estimator.window & custom_mask
    assert np.array_equal(
        result, expected_result
    ), "apply_mask did not apply the mask correctly."


def test_apply_mask_with_zeroed_mask():
    """Test apply_mask when a zeroed mask is provided."""
    img_dim = (10, 10)
    estimator = NoiseEstimator(img_dim, win_rad=3)

    # Create a mask where all pixels are excluded (False)
    zeroed_mask = np.zeros(img_dim, dtype=bool)

    result = estimator.apply_mask(zeroed_mask)

    # The result should be an array of False values, as the mask excludes all pixels
    expected_result = np.zeros(img_dim, dtype=bool)
    assert np.array_equal(
        result, expected_result
    ), "apply_mask did not handle the zeroed mask correctly."


def test_unobscured_zernike_projection():
    n_zernikes = 20
    wfe_dim = 256
    tol = 1e-1

    # Create zernike basis
    zernikes = zernike_generator(n_zernikes=n_zernikes, wfe_dim=wfe_dim)
    np_zernike_cube = np.zeros(
        (len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1])
    )
    for it in range(len(zernikes)):
        np_zernike_cube[it, :, :] = zernikes[it]
    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    # Create random zernike coefficient array
    zk_array = np.random.randn(1, n_zernikes, 1, 1)
    tf_zk_array = tf.convert_to_tensor(zk_array, dtype=tf.float32)

    # Compute OPD
    tf_unobscured_opd = tf.math.reduce_sum(
        tf.math.multiply(tf_zernike_cube, tf_zk_array), axis=1
    )

    # Compute normalisation factor
    norm_factor = compute_unobscured_zernike_projection(
        tf_zernike_cube[0, :, :], tf_zernike_cube[0, :, :]
    )

    # Compute projections for each zernike
    estimated_zk_array = np.array(
        [
            compute_unobscured_zernike_projection(
                tf_unobscured_opd, tf_zernike_cube[j, :, :], norm_factor=norm_factor
            )
            for j in range(n_zernikes)
        ]
    )

    rmse_error = np.linalg.norm(estimated_zk_array - zk_array[0, :, 0, 0])

    assert rmse_error < tol


def test_tf_decompose_obscured_opd_basis():
    n_zernikes = 20
    wfe_dim = 256
    tol = 1e-5

    # Create zernike basis
    zernikes = zernike_generator(n_zernikes=n_zernikes, wfe_dim=wfe_dim)
    np_zernike_cube = np.zeros(
        (len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1])
    )
    for it in range(len(zernikes)):
        np_zernike_cube[it, :, :] = zernikes[it]
    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    # Create obscurations
    obscurations = PSFSimulator.generate_euclid_pupil_obscurations(
        N_pix=wfe_dim, N_filter=2
    )
    tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.float32)

    # Create random zernike coefficient array
    zk_array = np.random.randn(1, n_zernikes, 1, 1)
    tf_zk_array = tf.convert_to_tensor(zk_array, dtype=tf.float32)

    # Compute OPD
    tf_unobscured_opd = tf.math.reduce_sum(
        tf.math.multiply(tf_zernike_cube, tf_zk_array), axis=1
    )
    # Obscure the OPD
    tf_obscured_opd = tf.math.multiply(
        tf_unobscured_opd, tf.expand_dims(tf_obscurations, axis=0)
    )

    # Compute zernike array from OPD
    obsc_coeffs = decompose_tf_obscured_opd_basis(
        tf_opd=tf_obscured_opd,
        tf_obscurations=tf_obscurations,
        tf_zk_basis=tf_zernike_cube,
        n_zernike=n_zernikes,
        iters=100,
    )

    rmse_error = np.linalg.norm(obsc_coeffs - zk_array[0, :, 0, 0])

    assert rmse_error < tol
