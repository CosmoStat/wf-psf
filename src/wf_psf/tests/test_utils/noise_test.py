import pytest
import numpy as np
from wf_psf.utils.noise import NoiseEstimator


def test_initialization():
    """Test if NoiseEstimator initializes correctly."""
    img_dim = (50, 50)
    win_rad = 10
    estimator = NoiseEstimator(img_dim, win_rad)

    assert estimator.img_dim == img_dim
    assert estimator.win_rad == win_rad
    assert isinstance(estimator.window, np.ndarray)
    assert estimator.window.shape == img_dim


def test_default_win_rad():
    """Test that the default_win_rad derives an int radius from the image dimensions."""
    assert NoiseEstimator.default_win_rad((32, 32)) == 10
    assert NoiseEstimator.default_win_rad((40, 40)) == 13
    assert NoiseEstimator.default_win_rad((100, 76)) == 31
    assert isinstance(NoiseEstimator.default_win_rad((32, 32)), int)


def test_initialization_uses_default_win_rad_when_omitted():
    """Test that omitting win_rad falls back to default_win_rad and produces the same window."""
    img_dim = (50, 50)
    estimator = NoiseEstimator(img_dim)
    explicit_estimator = NoiseEstimator(
        img_dim, NoiseEstimator.default_win_rad(img_dim)
    )

    assert estimator.win_rad == NoiseEstimator.default_win_rad(img_dim)
    assert isinstance(estimator.win_rad, int)
    np.testing.assert_array_equal(estimator.window, explicit_estimator.window)


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
    rng = np.random.default_rng(0)
    image = rng.normal(0, 10, img_dim)

    noise_estimation = estimator.estimate_noise(image)

    # The estimated noise should be close to 10 (the true std)
    assert np.isclose(noise_estimation, 10, atol=2)


def test_estimate_noise_with_custom_mask():
    """Test noise estimation with a custom mask applied outside the exclusion radius."""
    img_dim = (50, 50)
    estimator = NoiseEstimator(img_dim, win_rad=5)

    # Create synthetic noise with std=5
    rng = np.random.default_rng(0)
    image = rng.normal(0, 5, img_dim)

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
    assert np.array_equal(result, estimator.window), (
        "apply_mask should return the window when mask is None."
    )


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
    assert np.array_equal(result, expected_result), (
        "apply_mask did not apply the mask correctly."
    )


def test_apply_mask_with_zeroed_mask():
    """Test apply_mask when a zeroed mask is provided."""
    img_dim = (10, 10)
    estimator = NoiseEstimator(img_dim, win_rad=3)

    # Create a mask where all pixels are excluded (False)
    zeroed_mask = np.zeros(img_dim, dtype=bool)

    result = estimator.apply_mask(zeroed_mask)

    # The result should be an array of False values, as the mask excludes all pixels
    expected_result = np.zeros(img_dim, dtype=bool)
    assert np.array_equal(result, expected_result), (
        "apply_mask did not handle the zeroed mask correctly."
    )


def test_estimate_noise_with_fallback_outer_window_empty():
    """Outer-window selection is empty."""
    estimator = NoiseEstimator(img_dim=(30, 30))

    rng = np.random.default_rng(0)
    image = rng.normal(0, 3, size=(30, 30))
    expected_sigma = estimator.sigma_mad(image)

    estimator.window = np.zeros((30, 30), dtype=bool)
    sigma, fallback_used, total_pixels_used = estimator.estimate_noise_with_fallback(
        image
    )

    assert fallback_used is True
    assert total_pixels_used == image.size
    np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-4)


def test_estimate_noise_with_fallback_zero_mad():
    """Fallback is used when the outer-window estimate is zero."""
    estimator = NoiseEstimator(img_dim=(30, 30))

    image = np.zeros((30, 30))
    expected_sigma = estimator.sigma_mad(image)

    sigma, fallback_used, pixels_used = estimator.estimate_noise_with_fallback(image)

    assert estimator.window.any()
    assert fallback_used is True
    assert pixels_used == image.size
    assert sigma == expected_sigma


def test_estimate_noise_with_fallback_respects_mask():
    """Fallback uses all valid pixels, excluding masked pixels."""
    estimator = NoiseEstimator(img_dim=(30, 30))

    rng = np.random.default_rng(0)
    image = np.zeros((30, 30))
    mask = np.zeros((30, 30), dtype=bool)

    # Valid pixels have a non-zero MAD.
    mask[:10, :] = True
    image[:10, :] = rng.normal(0, 3, size=(10, 30))

    # Invalid pixels have a large value that would affect the estimate.
    image[10:, :] = 1000

    # Force the initial outer-window selection to be empty.
    estimator.window = np.zeros((30, 30), dtype=bool)

    expected_sigma = estimator.sigma_mad(image[mask])

    sigma, fallback_used, pixels_used = estimator.estimate_noise_with_fallback(
        image, mask
    )

    assert fallback_used is True
    assert pixels_used == mask.sum()
    np.testing.assert_allclose(sigma, expected_sigma)


def test_estimate_noise_with_fallback_no_valid_pixels():
    """Returns NaN when no valid pixels are available for fallback."""
    estimator = NoiseEstimator(img_dim=(30, 30))

    image = np.zeros((30, 30))
    mask = np.zeros((30, 30), dtype=bool)

    sigma, fallback_used, pixels_used = estimator.estimate_noise_with_fallback(
        image, mask
    )

    assert fallback_used is True
    assert pixels_used == 0
    assert np.isnan(sigma)


def test_estimate_noise_batch_without_masks():
    """estimate_noise_batch with no masks matches per-image estimate_noise."""
    img_dim = (30, 30)
    win_rad = 5
    estimator = NoiseEstimator(img_dim, win_rad)

    rng = np.random.default_rng(0)
    images = rng.normal(0, 3, size=(4, *img_dim))

    batch_result = estimator.estimate_noise_batch(images)
    expected = np.array([estimator.estimate_noise(img) for img in images])

    assert batch_result.shape == (4,)
    np.testing.assert_allclose(batch_result, expected)


def test_estimate_noise_batch_with_masks():
    """estimate_noise_batch with per-image masks matches per-image estimate_noise."""
    img_dim = (30, 30)
    win_rad = 5
    estimator = NoiseEstimator(img_dim, win_rad)

    rng = np.random.default_rng(0)
    images = rng.normal(0, 3, size=(4, *img_dim))
    masks = rng.integers(0, 2, size=(4, *img_dim)).astype(bool)

    batch_result = estimator.estimate_noise_batch(images, masks)
    expected = np.array(
        [estimator.estimate_noise(img, mask) for img, mask in zip(images, masks)]
    )

    assert batch_result.shape == (4,)
    np.testing.assert_allclose(batch_result, expected)


def test_estimate_noise_with_fallback_batch():
    """Batch API handles images that do and do not need fallback."""
    estimator = NoiseEstimator(img_dim=(30, 30))

    rng = np.random.default_rng(0)
    images = rng.normal(0, 3, size=(2, 30, 30))
    images[0] = 1.0  # Zero MAD: triggers fallback.

    masks = np.ones((2, 30, 30), dtype=bool)
    masks[0, :10, :] = False  # Exclude some pixels from the first image.

    sigmas, fallbacks, pixels_used = estimator.estimate_noise_batch_with_fallback(
        images, masks
    )

    expected_sigma_0 = estimator.sigma_mad(images[0][masks[0]])
    expected_sigma_1 = estimator.sigma_mad(images[1][estimator.window & masks[1]])

    np.testing.assert_array_equal(fallbacks, [True, False])
    np.testing.assert_allclose(sigmas, [expected_sigma_0, expected_sigma_1])
    np.testing.assert_array_equal(
        pixels_used,
        [masks[0].sum(), (estimator.window & masks[1]).sum()],
    )

    assert sigmas.shape == (2,)
    assert fallbacks.shape == (2,)
    assert fallbacks.dtype == bool
    assert pixels_used.shape == (2,)


@pytest.mark.parametrize(
    "method_name",
    ["estimate_noise_batch", "estimate_noise_batch_with_fallback"],
)
def test_estimate_noise_batch_rejects_images_with_wrong_dimensions(method_name):
    """estimate_noise_batch rejects images that are not 3D."""
    estimator = NoiseEstimator(img_dim=(30, 30))

    images = np.random.normal(0, 3, size=(4, 30, 30, 1))
    method = getattr(estimator, method_name)

    with pytest.raises(ValueError, match="images must have shape"):
        method(images)


@pytest.mark.parametrize(
    "method_name",
    ["estimate_noise_batch", "estimate_noise_batch_with_fallback"],
)
def test_estimate_noise_batch_rejects_images_with_wrong_image_shape(method_name):
    """estimate_noise_batch rejects images with unexpected spatial dimensions."""
    estimator = NoiseEstimator(img_dim=(30, 30))

    images = np.random.normal(0, 3, size=(4, 32, 30))

    method = getattr(estimator, method_name)

    with pytest.raises(ValueError, match="Expected image dimensions"):
        method(images)


@pytest.mark.parametrize(
    "method_name",
    ["estimate_noise_batch", "estimate_noise_batch_with_fallback"],
)
def test_estimate_noise_batch_rejects_masks_with_wrong_shape(method_name):
    """estimate_noise_batch rejects masks that do not match the image batch."""
    estimator = NoiseEstimator(img_dim=(30, 30))

    images = np.random.normal(0, 3, size=(4, 30, 30))
    masks = np.ones((4, 30, 29), dtype=bool)

    method = getattr(estimator, method_name)

    with pytest.raises(ValueError, match="masks and images must have matching shapes"):
        method(images, masks)


@pytest.mark.parametrize(
    "method_name",
    ["estimate_noise_batch", "estimate_noise_batch_with_fallback"],
)
def test_estimate_noise_batch_rejects_masks_with_wrong_batch_size(method_name):
    """estimate_noise_batch rejects masks with a different batch size."""
    estimator = NoiseEstimator(img_dim=(30, 30))

    images = np.random.normal(0, 3, size=(4, 30, 30))
    masks = np.ones((3, 30, 30), dtype=bool)

    method = getattr(estimator, method_name)
    with pytest.raises(ValueError, match="masks and images must have matching shapes"):
        method(images, masks)
