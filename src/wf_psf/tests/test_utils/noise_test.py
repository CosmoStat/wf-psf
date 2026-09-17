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
    explicit_estimator = NoiseEstimator(img_dim, NoiseEstimator.default_win_rad(img_dim))

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


def test_estimate_noise_batch_without_masks():
    """estimate_noise_batch with no masks matches per-image estimate_noise."""
    img_dim = (30, 30)
    win_rad = 5
    estimator = NoiseEstimator(img_dim, win_rad)

    np.random.seed(0)
    images = np.random.normal(0, 3, size=(4, *img_dim))

    batch_result = estimator.estimate_noise_batch(images)
    expected = np.array([estimator.estimate_noise(img) for img in images])

    assert batch_result.shape == (4,)
    np.testing.assert_allclose(batch_result, expected)


def test_estimate_noise_batch_with_masks():
    """estimate_noise_batch with per-image masks matches per-image estimate_noise."""
    img_dim = (30, 30)
    win_rad = 5
    estimator = NoiseEstimator(img_dim, win_rad)

    np.random.seed(1)
    images = np.random.normal(0, 3, size=(4, *img_dim))
    masks = np.random.randint(0, 2, size=(4, *img_dim)).astype(bool)

    batch_result = estimator.estimate_noise_batch(images, masks)
    expected = np.array(
        [estimator.estimate_noise(img, mask) for img, mask in zip(images, masks)]
    )

    assert batch_result.shape == (4,)
    np.testing.assert_allclose(batch_result, expected)
