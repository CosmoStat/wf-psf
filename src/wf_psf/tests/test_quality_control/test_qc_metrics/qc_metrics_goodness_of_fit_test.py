"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Metrics Goodness-of-Fit.

This module contains unit tests for the quality control metrics goodness-of-fit module.

:Author:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import numpy as np
import pytest
from unittest.mock import patch

from wf_psf.quality_control.context import QualityControlContext
from wf_psf.quality_control.metrics.goodness_of_fit import GoodnessOfFitMetric
from wf_psf.tests.test_quality_control.test_utils import get_pixel_mask
from wf_psf.utils.noise import NoiseEstimator


@pytest.fixture
def residuals():
    return np.array(
        [
            [[1.0, -2.0, 3.0], [-4.0, 0.0, 5.0]],
            [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]],
        ]
    )


@pytest.fixture
def valid_pixels():
    return np.array(
        [
            [[True, True, False], [True, False, True]],
            [[True, False, True], [False, True, True]],
        ]
    )


@pytest.fixture
def noise_sigma():
    return np.array([2.0, 3.0])


@pytest.fixture
def icovs():
    return np.array(
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        ]
    )


@pytest.fixture
def icovs_unequal():
    return np.array(
        [
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            [[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]],
        ]
    )


@pytest.fixture
def compute_context(dataset_factory):
    rng = np.random.default_rng(42)

    n_src, n_pix = 2, 3
    dataset = dataset_factory(n_src=n_src, n_pix=n_pix)

    shape = (n_src, n_pix, n_pix)
    observations = rng.normal(loc=10.0, scale=1.0, size=shape)
    residuals = rng.normal(loc=0.0, scale=0.5, size=shape)
    predictions = observations - residuals

    dataset.sources = observations
    dataset.masks[0] = get_pixel_mask(
        shape=(n_pix, n_pix),
        masked_pixels=[(0, 0), (1, 1)],
    )
    dataset.masks[1] = get_pixel_mask(
        shape=(n_pix, n_pix),
        masked_pixels=[(2, 2)],
    )

    context = QualityControlContext(dataset=dataset)
    context.resources["psf_models.standard"] = predictions

    return context, residuals


def test_validate_dataset_inputs_valid():
    observations = np.ones((2, 2, 3))
    predictions = np.ones((2, 2, 3))
    masks = np.zeros((2, 2, 3), dtype=bool)

    GoodnessOfFitMetric._validate_dataset_inputs(
        observations=observations, predictions=predictions, masks=masks
    )


@pytest.mark.parametrize(
    "observations",
    [
        np.ones((2, 3)),  # 2D
        np.ones((2, 2, 3, 4)),  # 4D
    ],
)
def test_validate_dataset_inputs_rejects_observations_with_wrong_dimensions(
    observations,
):
    predictions = np.ones((2, 2, 3))
    masks = np.zeros((2, 2, 3), dtype=bool)

    with pytest.raises(
        ValueError,
        match="observations must have shape",
    ):
        GoodnessOfFitMetric._validate_dataset_inputs(
            observations=observations,
            predictions=predictions,
            masks=masks,
        )


def test_validate_dataset_inputs_rejects_observations_and_predictions_with_nonmatching_shapes():
    observations = np.ones((3, 2, 3))
    predictions = np.ones((2, 2, 3))
    masks = np.zeros((2, 2, 3), dtype=bool)

    with pytest.raises(
        ValueError,
        match="observations and predictions must have matching shapes",
    ):
        GoodnessOfFitMetric._validate_dataset_inputs(
            observations=observations,
            predictions=predictions,
            masks=masks,
        )


def test_validate_dataset_inputs_rejects_observations_and_masks_with_nonmatching_shapes():
    observations = np.ones((3, 2, 3))
    predictions = np.ones((3, 2, 3))
    masks = np.zeros((2, 2, 3), dtype=bool)

    with pytest.raises(
        ValueError,
        match="observations and masks must have matching shapes",
    ):
        GoodnessOfFitMetric._validate_dataset_inputs(
            observations=observations,
            predictions=predictions,
            masks=masks,
        )


def test_validate_residual_inputs_valid(residuals, valid_pixels):
    GoodnessOfFitMetric._validate_residual_inputs(
        residuals=residuals, valid_pixels=valid_pixels
    )


@pytest.mark.parametrize(
    "residuals",
    [
        np.ones((2, 3)),  # 2D
        np.ones((2, 2, 3, 4)),  # 4D
    ],
)
def test_validate_residuals_rejects_residuals_with_wrong_dimensions(
    residuals, valid_pixels
):
    with pytest.raises(ValueError, match="residuals must have shape"):
        GoodnessOfFitMetric._validate_residual_inputs(
            residuals=residuals, valid_pixels=valid_pixels
        )


def test_validate_residual_inputs_rejects_nonmatching_shapes():
    residuals = np.ones((2, 2, 3))

    valid_pixels = np.ones((2, 2, 4), dtype=bool)

    with pytest.raises(
        ValueError,
        match="residuals and valid_pixels must have matching shapes",
    ):
        GoodnessOfFitMetric._validate_residual_inputs(
            residuals=residuals,
            valid_pixels=valid_pixels,
        )


def test_compute_sum_squared_residuals_valid(residuals, valid_pixels):
    expected = np.array([46.0, 71.0])

    result = GoodnessOfFitMetric._compute_sum_squared_residuals(residuals, valid_pixels)

    np.testing.assert_array_equal(result, expected)


def test_compute_sum_squared_residuals_all_pixels_valid(residuals):
    valid_pixels = np.ones(residuals.shape, dtype=bool)

    expected = np.sum(residuals**2, axis=(1, 2))

    result = GoodnessOfFitMetric._compute_sum_squared_residuals(residuals, valid_pixels)

    np.testing.assert_array_equal(result, expected)


def test_compute_sum_squared_residuals_no_pixels_valid(residuals):
    valid_pixels = np.zeros(residuals.shape, dtype=bool)

    expected = np.zeros(residuals.shape[0])

    result = GoodnessOfFitMetric._compute_sum_squared_residuals(residuals, valid_pixels)

    np.testing.assert_array_equal(result, expected)


def test_compute_sum_squared_residuals_batch(residuals, valid_pixels):
    valid_pixels[1] = False

    expected = np.array([46.0, 0.0])

    result = GoodnessOfFitMetric._compute_sum_squared_residuals(residuals, valid_pixels)

    np.testing.assert_array_equal(result, expected)


# ----- pixel RMSE tests


def test_compute_pixel_rmse_valid(residuals, valid_pixels):
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    expected = np.array([np.sqrt(46.0 / 4.0), np.sqrt(71.0 / 4.0)])

    result = GoodnessOfFitMetric._compute_pixel_rmse(
        residuals=residuals, valid_pixels=valid_pixels, n_valid_pixels=n_valid_pixels
    )

    np.testing.assert_allclose(result, expected)


def test_compute_pixel_rmse_all_pixels_valid(residuals):
    valid_pixels = np.ones(residuals.shape, dtype=bool)

    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    expected = np.array([np.sqrt(55.0 / 6.0), np.sqrt(91.0 / 6.0)])

    result = GoodnessOfFitMetric._compute_pixel_rmse(
        residuals=residuals, valid_pixels=valid_pixels, n_valid_pixels=n_valid_pixels
    )

    np.testing.assert_allclose(result, expected)


def test_compute_pixel_rmse_no_pixels_valid(residuals):
    valid_pixels = np.zeros(residuals.shape, dtype=bool)

    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    expected = np.array([np.nan, np.nan])

    result = GoodnessOfFitMetric._compute_pixel_rmse(
        residuals=residuals, valid_pixels=valid_pixels, n_valid_pixels=n_valid_pixels
    )

    np.testing.assert_array_equal(result, expected)


def test_compute_pixel_rmse_batch(residuals, valid_pixels):
    valid_pixels = valid_pixels.copy()
    valid_pixels[1] = False
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    expected = np.array([np.sqrt(46.0 / 4.0), np.nan])

    result = GoodnessOfFitMetric._compute_pixel_rmse(
        residuals=residuals, valid_pixels=valid_pixels, n_valid_pixels=n_valid_pixels
    )

    np.testing.assert_allclose(result, expected, equal_nan=True)


# ----- chi2 MAD tests
def test_compute_chi2_mad_rejects_noise_sigma_wrong_shape(residuals, valid_pixels):
    noise_sigma = np.array([2.0, 3.0, 4.0])
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    with pytest.raises(
        ValueError, match="'noise_sigma' must contain one value per sample"
    ):
        _ = GoodnessOfFitMetric._compute_chi2_mad(
            residuals=residuals,
            noise_sigma=noise_sigma,
            valid_pixels=valid_pixels,
            n_valid_pixels=n_valid_pixels,
        )


def test_compute_chi2_mad_rejects_n_valid_pixels_wrong_shape(
    residuals, valid_pixels, noise_sigma
):
    n_valid_pixels = np.array([4, 4, 4])

    with pytest.raises(
        ValueError, match="'n_valid_pixels' must contain one value per sample"
    ):
        GoodnessOfFitMetric._compute_chi2_mad(
            residuals=residuals,
            noise_sigma=noise_sigma,
            valid_pixels=valid_pixels,
            n_valid_pixels=n_valid_pixels,
        )


@pytest.mark.parametrize("invalid_sigma", [np.nan, np.inf, 0.0, -1.0])
def test_compute_chi2_mad_returns_nan_for_invalid_noise_sigma(
    residuals, valid_pixels, noise_sigma, invalid_sigma
):
    noise_sigma = noise_sigma.copy()
    noise_sigma[1] = invalid_sigma
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    result = GoodnessOfFitMetric._compute_chi2_mad(
        residuals=residuals,
        noise_sigma=noise_sigma,
        valid_pixels=valid_pixels,
        n_valid_pixels=n_valid_pixels,
    )

    assert np.isfinite(result[0])
    assert np.isnan(result[1])


def test_compute_chi2_mad_metrics_valid(residuals, valid_pixels, noise_sigma):
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    expected_chi2 = np.array([46.0 / 2.0**2, 275 / 4.0 / 3.0**2])
    expected_reduced_chi2 = expected_chi2 / (n_valid_pixels - 1.0)

    chi2_mad, reduced_chi2_mad = GoodnessOfFitMetric._compute_chi2_mad_metrics(
        residuals=residuals,
        valid_pixels=valid_pixels,
        noise_sigma=noise_sigma,
        n_valid_pixels=n_valid_pixels,
    )

    np.testing.assert_allclose(chi2_mad, expected_chi2)
    np.testing.assert_allclose(reduced_chi2_mad, expected_reduced_chi2)


def test_compute_chi2_mad_metrics_all_pixels_valid(residuals, noise_sigma):
    valid_pixels = np.ones(residuals.shape, dtype=bool)
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    expected_chi2 = np.array([107.0 / 2.0 / 2.0**2, 179.0 / 2.0 / 3.0**2])
    expected_reduced_chi2 = expected_chi2 / (n_valid_pixels - 1.0)

    chi2_mad, reduced_chi2_mad = GoodnessOfFitMetric._compute_chi2_mad_metrics(
        residuals=residuals,
        valid_pixels=valid_pixels,
        noise_sigma=noise_sigma,
        n_valid_pixels=n_valid_pixels,
    )

    np.testing.assert_allclose(chi2_mad, expected_chi2)
    np.testing.assert_allclose(reduced_chi2_mad, expected_reduced_chi2)


def test_compute_chi2_mad_metrics_no_pixels_valid(residuals, noise_sigma):
    valid_pixels = np.zeros(residuals.shape, dtype=bool)
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    expected_chi2 = np.array([np.nan, np.nan])
    expected_reduced_chi2 = np.array([np.nan, np.nan])

    chi2_mad, reduced_chi2_mad = GoodnessOfFitMetric._compute_chi2_mad_metrics(
        residuals=residuals,
        valid_pixels=valid_pixels,
        noise_sigma=noise_sigma,
        n_valid_pixels=n_valid_pixels,
    )

    np.testing.assert_allclose(chi2_mad, expected_chi2, equal_nan=True)
    np.testing.assert_allclose(reduced_chi2_mad, expected_reduced_chi2, equal_nan=True)


def test_compute_chi2_mad_one_pixel_valid(residuals, noise_sigma):
    valid_pixels = np.zeros(residuals.shape, dtype=bool)
    valid_pixels[:, 0, 0] = True
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    expected_chi2 = np.array([0.0, 0.0])
    expected_reduced_chi2 = np.array([np.nan, np.nan])

    chi2_mad, reduced_chi2_mad = GoodnessOfFitMetric._compute_chi2_mad_metrics(
        residuals=residuals,
        valid_pixels=valid_pixels,
        noise_sigma=noise_sigma,
        n_valid_pixels=n_valid_pixels,
    )

    np.testing.assert_allclose(chi2_mad, expected_chi2, equal_nan=True)
    np.testing.assert_allclose(reduced_chi2_mad, expected_reduced_chi2, equal_nan=True)


def test_compute_chi2_mad_metrics_batch(residuals, valid_pixels, noise_sigma):
    valid_pixels[1] = False
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    chi2_mad, reduced_chi2_mad = GoodnessOfFitMetric._compute_chi2_mad_metrics(
        residuals=residuals,
        valid_pixels=valid_pixels,
        noise_sigma=noise_sigma,
        n_valid_pixels=n_valid_pixels,
    )

    np.testing.assert_allclose(chi2_mad, [46.0 / 2.0**2, np.nan], equal_nan=True)
    np.testing.assert_allclose(
        reduced_chi2_mad,
        [46.0 / 2.0**2 / 3.0, np.nan],
        equal_nan=True,
    )


def test_compute_chi2_mad_metrics_propagates_invalid_noise_sigma(
    residuals, valid_pixels, noise_sigma
):
    noise_sigma[1] = np.nan
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    chi2_mad, reduced_chi2_mad = GoodnessOfFitMetric._compute_chi2_mad_metrics(
        residuals=residuals,
        valid_pixels=valid_pixels,
        noise_sigma=noise_sigma,
        n_valid_pixels=n_valid_pixels,
    )

    assert np.isfinite(chi2_mad[0])
    assert np.isnan(chi2_mad[1])
    assert np.isfinite(reduced_chi2_mad[0])
    assert np.isnan(reduced_chi2_mad[1])


@pytest.mark.parametrize(
    ("noise_sigma", "n_pixels"),
    [
        (0.1, 16**2),
        (0.5, 32**2),
        (2.0, 64**2),
    ],
)
def test_compute_chi2_mad_with_varying_noise_sigma(noise_sigma, n_pixels):
    # Generate Gaussian residuals with different standard deviations across
    # samples, but a constant standard deviation across pixels within each sample.
    # Since the residuals are generated using the same noise sigma used to
    # calculate chi-square, the expected mean reduced chi-square is close to 1,
    # regardless of the noise scale.
    rng = np.random.default_rng(42)

    n_samples = 100
    noise_sigma = np.full(n_samples, noise_sigma)

    residuals = rng.normal(
        loc=0.0,
        scale=noise_sigma[:, None, None],
        size=(n_samples, 1, n_pixels),
    )

    valid_pixels = np.ones(residuals.shape, dtype=bool)
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    chi2_mad, reduced_chi2_mad = GoodnessOfFitMetric._compute_chi2_mad_metrics(
        residuals=residuals,
        valid_pixels=valid_pixels,
        noise_sigma=noise_sigma,
        n_valid_pixels=n_valid_pixels,
    )

    assert np.all(np.isfinite(chi2_mad))
    assert np.all(np.isfinite(reduced_chi2_mad))
    assert np.mean(reduced_chi2_mad) == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize(
    ("noise_sigma", "n_pixels", "offset"),
    [
        (0.1, 16**2, -3.5),
        (0.5, 32**2, 2.5),
        (2.0, 64**2, 3.5),
    ],
)
def test_compute_chi2_mad_constant_offset_invariance(noise_sigma, n_pixels, offset):
    # Generate Gaussian residuals with different standard deviations across
    # samples, but a constant standard deviation across pixels within each sample.
    # Adding the same constant offset to every pixel should not change chi-square
    # because the sample mean residual is subtracted.
    rng = np.random.default_rng(42)

    n_samples = 100
    noise_sigma = np.full(n_samples, noise_sigma)

    residuals = rng.normal(
        loc=0.0,
        scale=noise_sigma[:, None, None],
        size=(n_samples, 1, n_pixels),
    )

    valid_pixels = np.ones(residuals.shape, dtype=bool)
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    chi2_mad, reduced_chi2_mad = GoodnessOfFitMetric._compute_chi2_mad_metrics(
        residuals=residuals,
        valid_pixels=valid_pixels,
        noise_sigma=noise_sigma,
        n_valid_pixels=n_valid_pixels,
    )

    # add offset to residuals
    offset_residuals = residuals + offset

    offset_chi2_mad, offset_reduced_chi2_mad = (
        GoodnessOfFitMetric._compute_chi2_mad_metrics(
            residuals=offset_residuals,
            valid_pixels=valid_pixels,
            noise_sigma=noise_sigma,
            n_valid_pixels=n_valid_pixels,
        )
    )

    np.testing.assert_allclose(chi2_mad, offset_chi2_mad, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        reduced_chi2_mad, offset_reduced_chi2_mad, rtol=1e-12, atol=1e-12
    )


# ----- chi2 ICOV tests
def test_compute_weighted_mean_residual_all_pixels_usable(residuals, icovs):
    usable_pixels = np.ones(residuals.shape, dtype=bool)

    expected_mean_residual = np.array([0.5, 0.5])
    arithmetic_mean = np.mean(residuals, axis=(1, 2))

    mean_residual = GoodnessOfFitMetric._compute_weighted_mean_residual(
        residuals=residuals, usable_pixels=usable_pixels, icovs=icovs
    )

    np.testing.assert_allclose(mean_residual, expected_mean_residual)
    np.testing.assert_allclose(mean_residual, arithmetic_mean)


def test_compute_weighted_mean_residual_unequal_icovs(residuals, icovs_unequal):
    usable_pixels = np.ones(residuals.shape, dtype=bool)

    expected_mean_residual = np.array(
        [
            17.0 / 12.0,
            3.0 / 6.0,
        ]
    )

    mean_residual = GoodnessOfFitMetric._compute_weighted_mean_residual(
        residuals=residuals,
        usable_pixels=usable_pixels,
        icovs=icovs_unequal,
    )

    np.testing.assert_allclose(mean_residual, expected_mean_residual)


def test_compute_weighted_mean_residual_valid(residuals, valid_pixels, icovs_unequal):
    usable_pixels = valid_pixels & np.isfinite(icovs_unequal) & (icovs_unequal > 0)

    mean_residual = GoodnessOfFitMetric._compute_weighted_mean_residual(
        residuals=residuals,
        usable_pixels=usable_pixels,
        icovs=icovs_unequal,
    )

    expected_mean_residual = np.array([8.0 / 7.0, -1.0 / 4.5])

    np.testing.assert_allclose(mean_residual, expected_mean_residual)


def test_compute_weighted_mean_residual_no_pixels_usable(residuals, icovs_unequal):
    usable_pixels = np.zeros(residuals.shape, dtype=bool)

    mean_residual = GoodnessOfFitMetric._compute_weighted_mean_residual(
        residuals=residuals,
        usable_pixels=usable_pixels,
        icovs=icovs_unequal,
    )

    expected_mean_residual = np.array([np.nan, np.nan])

    np.testing.assert_allclose(mean_residual, expected_mean_residual, equal_nan=True)


@pytest.mark.parametrize("invalid_icov", [0.0, -1.0, np.nan, np.inf])
def test_compute_weighted_mean_residual_invalid_icov(
    invalid_icov, residuals, valid_pixels, icovs_unequal
):
    icovs_unequal[1] = invalid_icov
    usable_pixels = valid_pixels & np.isfinite(icovs_unequal) & (icovs_unequal > 0)

    expected_mean_residual = np.array([8.0 / 7.0, np.nan])

    mean_residual = GoodnessOfFitMetric._compute_weighted_mean_residual(
        residuals=residuals,
        usable_pixels=usable_pixels,
        icovs=icovs_unequal,
    )

    np.testing.assert_allclose(mean_residual, expected_mean_residual, equal_nan=True)


@pytest.mark.parametrize("invalid_icov", [np.nan, np.inf, 0.0, -1.0])
def test_compute_chi2_icov_excludes_invalid_icov(
    residuals, valid_pixels, icovs, invalid_icov
):
    icovs = icovs.copy()
    icovs[1] = invalid_icov

    expected_chi2 = np.array([46.0, 0.0])
    expected_n_valid_icov_pixels = np.array([4, 0])

    chi2_icov, n_valid_icov_pixels = GoodnessOfFitMetric._compute_chi2_icov(
        residuals=residuals,
        valid_pixels=valid_pixels,
        icovs=icovs,
    )

    np.testing.assert_allclose(chi2_icov, expected_chi2)
    np.testing.assert_allclose(n_valid_icov_pixels, expected_n_valid_icov_pixels)


def test_compute_chi2_icov_metrics_valid(residuals, valid_pixels, icovs):
    expected_chi2 = np.array([46.0, 34.375])
    expected_reduced_chi2 = np.array([46.0 / 3.0, 34.375 / 3.0])
    expected_n_valid_icov_pixels = np.array([4, 4])

    chi2_icov, reduced_chi2_icov, n_valid_icov_pixels = (
        GoodnessOfFitMetric._compute_chi2_icov_metrics(
            residuals=residuals, valid_pixels=valid_pixels, icovs=icovs
        )
    )

    np.testing.assert_allclose(chi2_icov, expected_chi2)
    np.testing.assert_allclose(reduced_chi2_icov, expected_reduced_chi2)
    np.testing.assert_allclose(n_valid_icov_pixels, expected_n_valid_icov_pixels)


def test_compute_chi2_icov_all_pixels_valid(residuals, icovs):
    valid_pixels = np.ones(residuals.shape, dtype=bool)

    expected_chi2 = np.array([53.5, 44.75])
    expected_reduced_chi2 = np.array([53.5 / 5.0, 44.75 / 5.0])
    expected_n_valid_icov_pixels = np.array([6, 6])

    chi2_icov, reduced_chi2_icov, n_valid_icov_pixels = (
        GoodnessOfFitMetric._compute_chi2_icov_metrics(
            residuals=residuals, valid_pixels=valid_pixels, icovs=icovs
        )
    )

    np.testing.assert_allclose(chi2_icov, expected_chi2)
    np.testing.assert_allclose(reduced_chi2_icov, expected_reduced_chi2)
    np.testing.assert_allclose(n_valid_icov_pixels, expected_n_valid_icov_pixels)


def test_compute_chi2_icov_no_pixels_valid(residuals, icovs):
    valid_pixels = np.zeros(residuals.shape, dtype=bool)

    expected_chi2 = np.array([0.0, 0.0])
    expected_reduced_chi2 = np.array([np.nan, np.nan])
    expected_n_valid_icov_pixels = np.array([0, 0])

    chi2_icov, reduced_chi2_icov, n_valid_icov_pixels = (
        GoodnessOfFitMetric._compute_chi2_icov_metrics(
            residuals=residuals, valid_pixels=valid_pixels, icovs=icovs
        )
    )

    np.testing.assert_allclose(chi2_icov, expected_chi2)
    np.testing.assert_allclose(reduced_chi2_icov, expected_reduced_chi2, equal_nan=True)
    np.testing.assert_allclose(n_valid_icov_pixels, expected_n_valid_icov_pixels)


def test_compute_chi2_icov_one_pixel_valid(residuals, icovs):
    valid_pixels = np.zeros(residuals.shape, dtype=bool)
    valid_pixels[:, 0, 0] = True

    expected_chi2 = np.array([0.0, 0.0])
    expected_reduced_chi2 = np.array([np.nan, np.nan])
    expected_n_valid_icov_pixels = np.array([1, 1])

    chi2_icov, reduced_chi2_icov, n_valid_icov_pixels = (
        GoodnessOfFitMetric._compute_chi2_icov_metrics(
            residuals=residuals, valid_pixels=valid_pixels, icovs=icovs
        )
    )

    np.testing.assert_allclose(chi2_icov, expected_chi2)
    np.testing.assert_allclose(reduced_chi2_icov, expected_reduced_chi2, equal_nan=True)
    np.testing.assert_allclose(n_valid_icov_pixels, expected_n_valid_icov_pixels)


def test_compute_chi2_icov_batch(residuals, valid_pixels, icovs):
    valid_pixels = valid_pixels.copy()
    valid_pixels[1] = False

    expected_chi2 = np.array([46.0, 0.0])
    expected_reduced_chi2 = np.array([46.0 / 3.0, np.nan])
    expected_n_valid_icov_pixels = np.array([4, 0])

    chi2_icov, reduced_chi2_icov, n_valid_icov_pixels = (
        GoodnessOfFitMetric._compute_chi2_icov_metrics(
            residuals=residuals, valid_pixels=valid_pixels, icovs=icovs
        )
    )

    np.testing.assert_allclose(chi2_icov, expected_chi2)
    np.testing.assert_allclose(reduced_chi2_icov, expected_reduced_chi2)
    np.testing.assert_allclose(n_valid_icov_pixels, expected_n_valid_icov_pixels)


def test_compute_chi2_icov_masked_pixels_not_counted(residuals, valid_pixels, icovs):
    expected = GoodnessOfFitMetric._compute_chi2_icov_metrics(
        residuals=residuals.copy(),
        valid_pixels=valid_pixels.copy(),
        icovs=icovs.copy(),
    )

    # Change values only at masked pixels.
    masked_pixels = ~valid_pixels
    modified_residuals = residuals.copy()
    modified_icovs = icovs.copy()

    modified_residuals[masked_pixels] = 1e6
    modified_icovs[masked_pixels] = 1e6

    result = GoodnessOfFitMetric._compute_chi2_icov_metrics(
        residuals=modified_residuals,
        valid_pixels=valid_pixels,
        icovs=modified_icovs,
    )

    for actual, reference in zip(result, expected):
        np.testing.assert_allclose(actual, reference, equal_nan=True)

    np.testing.assert_array_equal(result[2], [4, 4])


def test_compute_chi2_icov_rejects_icov_wrong_shape(residuals, valid_pixels):
    icovs = np.array(
        [
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ]
    )
    with pytest.raises(
        ValueError, match="'icovs' must have the same shape as 'residuals'"
    ):
        _ = GoodnessOfFitMetric._compute_chi2_icov(
            residuals=residuals, valid_pixels=valid_pixels, icovs=icovs
        )


@pytest.mark.parametrize(
    ("noise_sigma", "n_pixels"),
    [
        (0.1, 16),
        (0.5, 32),
        (2.0, 64),
    ],
)
def test_compute_chi2_icov_with_spatially_varying_inverse_variance(
    noise_sigma, n_pixels
):
    # Generate Gaussian residuals with per-pixel standard deviations
    # corresponding to the inverse-variance map. The reduced chi-square
    # should therefore have an expected value close to 1.
    rng = np.random.default_rng(42)

    n_samples = 100
    noise_sigma = np.full(n_samples, noise_sigma)

    icovs = rng.uniform(
        low=0.5 / noise_sigma[0] ** 2,
        high=2.0 / noise_sigma[0] ** 2,
        size=(n_samples, n_pixels, n_pixels),
    )

    residuals = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(icovs),
    )

    valid_pixels = np.ones((n_samples, n_pixels, n_pixels), dtype=bool)

    expected_n_valid_icov_pixels = np.sum(valid_pixels, axis=(1, 2))

    chi2_icov, reduced_chi2_icov, n_valid_icov_pixels = (
        GoodnessOfFitMetric._compute_chi2_icov_metrics(
            residuals=residuals,
            icovs=icovs,
            valid_pixels=valid_pixels,
        )
    )

    assert np.all(np.isfinite(chi2_icov))
    assert np.all(np.isfinite(reduced_chi2_icov))
    assert np.mean(reduced_chi2_icov) == pytest.approx(1.0, abs=0.02)
    np.testing.assert_array_equal(n_valid_icov_pixels, expected_n_valid_icov_pixels)


@pytest.mark.parametrize(
    ("noise_sigma", "n_pixels", "offset"),
    [
        (0.1, 16, -3.5),
        (0.5, 32, 0.0),
        (2.0, 64, 3.5),
    ],
)
def test_compute_chi2_icov_offset_invariance_with_spatially_varying_icov(
    noise_sigma, n_pixels, offset
):
    # Generate Gaussian residuals with per-pixel standard deviations
    # corresponding to the spatially varying inverse-variance map.
    # Adding a constant offset should not change chi-square because
    # the weighted mean residual is subtracted.
    rng = np.random.default_rng(42)

    n_samples = 100
    noise_sigma = np.full(n_samples, noise_sigma)

    icovs = rng.uniform(
        low=0.5 / noise_sigma[0] ** 2,
        high=2.0 / noise_sigma[0] ** 2,
        size=(n_samples, n_pixels, n_pixels),
    )

    residuals = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(icovs),
    )

    valid_pixels = np.ones((n_samples, n_pixels, n_pixels), dtype=bool)

    chi2_icov, reduced_chi2_icov, _ = GoodnessOfFitMetric._compute_chi2_icov_metrics(
        residuals=residuals,
        icovs=icovs,
        valid_pixels=valid_pixels,
    )

    offset_residuals = residuals + offset

    offset_chi2_icov, offset_reduced_chi2_icov, _ = (
        GoodnessOfFitMetric._compute_chi2_icov_metrics(
            residuals=offset_residuals,
            icovs=icovs,
            valid_pixels=valid_pixels,
        )
    )

    np.testing.assert_allclose(chi2_icov, offset_chi2_icov, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        reduced_chi2_icov, offset_reduced_chi2_icov, rtol=1e-12, atol=1e-12
    )


# ----- compute API tests


def test_compute_without_icovs(compute_context, noise_sigma):
    context, residuals = compute_context

    gof_metrics = GoodnessOfFitMetric(params={})

    valid_pixels = ~context.dataset.masks
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    expected_rmse = np.sqrt(
        np.sum(
            np.where(valid_pixels, residuals**2, 0.0),
            axis=(1, 2),
        )
        / n_valid_pixels
    )

    expected_mean_residual = (
        np.sum(
            np.where(valid_pixels, residuals, 0.0),
            axis=(1, 2),
        )
        / n_valid_pixels
    )

    expected_chi2_mad = np.sum(
        np.where(
            valid_pixels,
            (residuals - expected_mean_residual[:, None, None]) ** 2
            / noise_sigma[:, None, None] ** 2,
            0.0,
        ),
        axis=(1, 2),
    )

    expected_reduced_chi2_mad = expected_chi2_mad / (n_valid_pixels - 1)

    with patch.object(NoiseEstimator, "estimate_noise_batch", return_value=noise_sigma):
        results = gof_metrics.compute(context=context)
        assert results.keys() == {
            "pixel_rmse",
            "n_valid_pixels",
            "chi2_mad",
            "reduced_chi2_mad",
        }

        np.testing.assert_allclose(results["pixel_rmse"], expected_rmse)

        np.testing.assert_array_equal(results["n_valid_pixels"], n_valid_pixels)

        np.testing.assert_allclose(results["chi2_mad"], expected_chi2_mad)

        np.testing.assert_allclose(
            results["reduced_chi2_mad"], expected_reduced_chi2_mad
        )


def test_compute_with_icovs(compute_context, noise_sigma):
    context, residuals = compute_context

    rng = np.random.default_rng(123)
    context.dataset.icovs = rng.uniform(
        low=0.5,
        high=2.0,
        size=context.dataset.sources.shape,
    )
    context.dataset.icovs[0, 0, 1] = 0.0
    context.dataset.icovs[1, 1, 0] = np.nan

    gof_metrics = GoodnessOfFitMetric(params={})

    valid_pixels = ~context.dataset.masks
    n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))

    expected_rmse = np.sqrt(
        np.sum(
            np.where(valid_pixels, residuals**2, 0.0),
            axis=(1, 2),
        )
        / n_valid_pixels
    )

    expected_mean_residual = (
        np.sum(
            np.where(valid_pixels, residuals, 0.0),
            axis=(1, 2),
        )
        / n_valid_pixels
    )

    expected_chi2_mad = np.sum(
        np.where(
            valid_pixels,
            (residuals - expected_mean_residual[:, None, None]) ** 2
            / noise_sigma[:, None, None] ** 2,
            0.0,
        ),
        axis=(1, 2),
    )

    expected_reduced_chi2_mad = expected_chi2_mad / (n_valid_pixels - 1)

    valid_icov_pixels = (
        valid_pixels & np.isfinite(context.dataset.icovs) & (context.dataset.icovs > 0)
    )
    n_valid_icov_pixels = np.sum(valid_icov_pixels, axis=(1, 2))

    weighted_residual_sum = np.sum(
        np.where(valid_icov_pixels, residuals * context.dataset.icovs, 0.0),
        axis=(1, 2),
    )
    sum_icovs = np.sum(
        np.where(valid_icov_pixels, context.dataset.icovs, 0.0), axis=(1, 2)
    )

    expected_mean_weighted_residual = weighted_residual_sum / sum_icovs

    expected_chi2_icov = np.sum(
        np.where(
            valid_icov_pixels,
            (residuals - expected_mean_weighted_residual[:, None, None]) ** 2
            * context.dataset.icovs,
            0.0,
        ),
        axis=(1, 2),
    )

    expected_reduced_chi2_icov = expected_chi2_icov / (n_valid_icov_pixels - 1)

    with patch.object(NoiseEstimator, "estimate_noise_batch", return_value=noise_sigma):
        results = gof_metrics.compute(context=context)
        assert results.keys() == {
            "pixel_rmse",
            "n_valid_pixels",
            "chi2_mad",
            "reduced_chi2_mad",
            "chi2_icov",
            "reduced_chi2_icov",
            "n_valid_icov_pixels",
        }

        np.testing.assert_allclose(results["pixel_rmse"], expected_rmse)

        np.testing.assert_array_equal(results["n_valid_pixels"], n_valid_pixels)

        np.testing.assert_allclose(results["chi2_mad"], expected_chi2_mad)

        np.testing.assert_allclose(
            results["reduced_chi2_mad"], expected_reduced_chi2_mad
        )

        np.testing.assert_allclose(results["chi2_icov"], expected_chi2_icov)

        np.testing.assert_allclose(
            results["reduced_chi2_icov"], expected_reduced_chi2_icov
        )

        np.testing.assert_array_equal(
            results["n_valid_icov_pixels"], n_valid_icov_pixels
        )

        assert np.all(n_valid_icov_pixels < n_valid_pixels)


def test_compute_with_no_valid_pixels(compute_context, noise_sigma):
    context, _ = compute_context
    context.dataset.masks[0] = True

    rng = np.random.default_rng(123)
    context.dataset.icovs = rng.uniform(
        low=0.5,
        high=2.0,
        size=context.dataset.sources.shape,
    )

    gof_metrics = GoodnessOfFitMetric(params={})

    with (
        patch.object(
            NoiseEstimator,
            "estimate_noise_batch",
            return_value=noise_sigma,
        ),
        pytest.warns(
            UserWarning,
            match="Cannot compute goodness-of-fit metrics for 1 sample",
        ),
    ):
        results = gof_metrics.compute(context=context)

    assert results["n_valid_pixels"][0] == 0
    assert np.isnan(results["pixel_rmse"][0])
    assert np.isnan(results["chi2_mad"][0])
    assert np.isnan(results["reduced_chi2_mad"][0])
    assert results["n_valid_icov_pixels"][0] == 0
    assert results["chi2_icov"][0] == 0.0
    assert np.isnan(results["reduced_chi2_icov"][0])

    assert results["n_valid_pixels"][1] > 0
    assert np.isfinite(results["pixel_rmse"][1])
    assert np.isfinite(results["chi2_mad"][1])
    assert np.isfinite(results["reduced_chi2_mad"][1])
    assert results["n_valid_icov_pixels"][1] > 0
    assert np.isfinite(results["chi2_icov"][1])
    assert np.isfinite(results["reduced_chi2_icov"][1])
