"""Goodness-of-fit quality metric.

Provides methods for evaluating PSF model agreement with observations
using per-sample residual statistics, including pixel RMSE and
chi-square metrics based on MAD noise estimates or inverse covariance
maps.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from wf_psf.quality_control.context import QualityControlContext

from .base import QualityMetric
import numpy as np
from typing import ClassVar
import warnings

from wf_psf.utils.noise import NoiseEstimator


class GoodnessOfFitMetric(QualityMetric):
    """Compute per-sample goodness-of-fit diagnostics for PSF predictions."""

    name = "goodness_of_fit"
    diagnostics: ClassVar[frozenset[str]] = frozenset(
        {
            "pixel_rmse",
            "n_valid_pixels",
            "chi2_mad",
            "reduced_chi2_mad",
            "chi2_icov",
            "reduced_chi2_icov",
        }
    )

    def compute(self, context: QualityControlContext) -> dict[str, np.ndarray]:
        """Compute goodness-of-fit metrics for each dataset sample.

        Parameters
        ----------
        context : QualityControlContext
            Quality control context containing the dataset and resources required to evaluate the metric.

        Returns
        -------
        dict[str, np.ndarray]
            Per-sample goodness-of-fit diagnostics, including pixel RMSE, the number of valid pixels, and chi-square statistics.

        """
        observations = self._get_sources(context)
        img_dim = (observations.shape[-2], observations.shape[-1])
        masks = self._get_masks(context=context, image_shape=observations.shape)
        predictions = self._get_predictions(context)
        icovs = self._get_icovs(context)

        self._validate_dataset_inputs(
            observations=observations,
            predictions=predictions,
            masks=masks,
        )

        noise_estimator = NoiseEstimator(img_dim=img_dim)

        valid_pixels = ~masks
        n_valid_pixels, invalid = self._count_valid_pixels(valid_pixels)

        if np.any(invalid):
            warnings.warn(
                f"Cannot compute goodness-of-fit metrics for "
                f"{np.count_nonzero(invalid)} sample(s) with no valid pixels.",
                UserWarning,
                stacklevel=2,
            )

        residuals = observations - predictions

        pixel_rmse = self._compute_pixel_rmse(
            residuals=residuals,
            valid_pixels=valid_pixels,
            n_valid_pixels=n_valid_pixels,
        )

        chi2_mad, reduced_chi2_mad = self._compute_chi2_mad_metrics(
            residuals=residuals,
            valid_pixels=valid_pixels,
            noise_sigma=noise_estimator.estimate_noise_batch(
                images=observations, masks=masks
            ),
            n_valid_pixels=n_valid_pixels,
        )

        results = {
            "pixel_rmse": pixel_rmse,
            "n_valid_pixels": n_valid_pixels,
            "chi2_mad": chi2_mad,
            "reduced_chi2_mad": reduced_chi2_mad,
        }

        if icovs is not None:
            (
                chi2_icov,
                reduced_chi2_icov,
                n_valid_icov_pixels,
            ) = self._compute_chi2_icov_metrics(
                residuals=residuals,
                icovs=icovs,
                valid_pixels=valid_pixels,
            )

            results.update(
                {
                    "chi2_icov": chi2_icov,
                    "reduced_chi2_icov": reduced_chi2_icov,
                    "n_valid_icov_pixels": n_valid_icov_pixels,
                }
            )

        return results

    def _get_sources(self, context: QualityControlContext) -> np.ndarray:
        """Return the dataset sources."""
        return context.dataset.sources

    def _get_masks(
        self,
        context: QualityControlContext,
        image_shape: tuple[int, ...],
    ) -> np.ndarray:
        """Return dataset pixel masks, where True indicates an invalid pixel.

        If the dataset has no masks, return an all-False mask.

        Parameters
        ----------
        context : QualityControlContext
            Quality control context containing the dataset.

        image_shape : tuple[int, ...]
        Expected shape of the observation array, used to validate the mask shape.

        Returns
        -------
        np.ndarray
            Boolean mask array matching ``image_shape``. If the dataset has no masks,
            all pixels are considered valid.

        Raises
        ------
        ValueError
            If the dataset mask shape does not match ``image_shape``.
        """
        masks = getattr(context.dataset, "masks", None)

        if masks is None:
            return np.zeros(image_shape, dtype=bool)

        masks = np.asarray(masks, dtype=bool)

        if masks.shape != image_shape:
            raise ValueError(
                f"Dataset masks have shape {masks.shape}, "
                f"but observations have shape {image_shape}."
            )

        return masks

    def _get_icovs(self, context: QualityControlContext) -> np.ndarray | None:
        """Retrieve inverse covariance maps if available."""
        return getattr(context.dataset, "icovs", None)

    def _get_predictions(self, context: QualityControlContext) -> np.ndarray:
        """Return PSF model predictions."""
        return context.resources["psf_models.standard"]

    @staticmethod
    def _count_valid_pixels(
        valid_pixels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Count valid pixels per sample and identify samples with none."""
        n_valid_pixels = np.sum(valid_pixels, axis=(1, 2))
        invalid = n_valid_pixels == 0

        return n_valid_pixels, invalid

    @staticmethod
    def _validate_dataset_inputs(
        observations: np.ndarray, predictions: np.ndarray, masks: np.ndarray
    ) -> None:
        """Validate observation, prediction, and mask array dimensions and shapes.

        Raises
        ------
        ValueError
            If observations are not 3D or if predictions or masks have
            shapes that do not match observations.
        """
        if observations.ndim != 3:
            raise ValueError("observations must have shape (n_samples, height, width).")

        if predictions.shape != observations.shape:
            raise ValueError("observations and predictions must have matching shapes.")

        if masks.shape != observations.shape:
            raise ValueError("observations and masks must have matching shapes.")

    @staticmethod
    def _validate_residual_inputs(
        residuals: np.ndarray,
        valid_pixels: np.ndarray,
    ) -> None:
        """Validate residual and valid-pixel arrays.

        Raises
        ------
        ValueError
            If residuals are not 3D or if residuals or the valid_pixels have
            different shapes.
        """
        if residuals.ndim != 3:
            raise ValueError("residuals must have shape (n_samples, height, width).")

        if residuals.shape != valid_pixels.shape:
            raise ValueError("residuals and valid_pixels must have matching shapes.")

    @staticmethod
    def _compute_sum_squared_residuals(
        residuals: np.ndarray,
        valid_pixels: np.ndarray,
    ) -> np.ndarray:
        """Compute the sum of squared residuals over valid pixels per sample."""
        GoodnessOfFitMetric._validate_residual_inputs(
            residuals=residuals, valid_pixels=valid_pixels
        )
        return np.sum(
            np.where(valid_pixels, residuals**2, 0.0),
            axis=(1, 2),
        )

    @staticmethod
    def _compute_pixel_rmse(
        residuals: np.ndarray,
        valid_pixels: np.ndarray,
        n_valid_pixels: np.ndarray,
    ) -> np.ndarray:
        """Compute per-sample RMSE from residuals.

        Samples with no valid pixels receive NaN.

        Notes
        -----
        ``n_valid_pixels`` must contain the number of valid pixels per sample,
        as determined from ``valid_pixels``.
        """
        sum_squared_residuals = GoodnessOfFitMetric._compute_sum_squared_residuals(
            residuals=residuals, valid_pixels=valid_pixels
        )

        rmse = np.full(n_valid_pixels.shape, np.nan, dtype=float)

        valid_samples = n_valid_pixels > 0

        rmse[valid_samples] = np.sqrt(
            sum_squared_residuals[valid_samples] / n_valid_pixels[valid_samples]
        )

        return rmse

    @staticmethod
    def _compute_chi2_mad(
        residuals: np.ndarray,
        valid_pixels: np.ndarray,
        noise_sigma: np.ndarray,
        n_valid_pixels: np.ndarray,
    ) -> np.ndarray:
        """Compute per-sample chi-square using MAD noise estimates.

        The mean residual is subtracted from each sample's residuals before
        computing chi-square. The noise estimate is a single scalar per sample,
        applied uniformly to its valid pixels.

        Samples with no valid pixels or an invalid noise estimate (non-finite
        or non-positive) receive NaN.

        Parameters
        ----------
        residuals : np.ndarray
            Residual images with shape (n_samples, height, width).

        valid_pixels : np.ndarray
            Boolean mask identifying valid pixels, with True indicating a
            valid pixel.

        noise_sigma : np.ndarray
            MAD-based noise estimates, with one value per sample.

        n_valid_pixels : np.ndarray
            Number of valid pixels per sample, as determined from
            ``valid_pixels``.

        Returns
        -------
        np.ndarray
            Chi-square values, with one value per sample. Samples with
            unusable inputs receive NaN.

        Raises
        ------
        ValueError
            If residuals and valid_pixels have incompatible dimensions or
            shapes, or if noise_sigma or n_valid_pixels does not contain
            exactly one value per sample.
        """
        GoodnessOfFitMetric._validate_residual_inputs(
            residuals=residuals, valid_pixels=valid_pixels
        )

        if noise_sigma.shape != (residuals.shape[0],):
            raise ValueError("'noise_sigma' must contain one value per sample.")

        if n_valid_pixels.shape != (residuals.shape[0],):
            raise ValueError("'n_valid_pixels' must contain one value per sample.")

        chi2_mad = np.full(residuals.shape[0], np.nan, dtype=float)

        mean_residual = np.full(residuals.shape[0], np.nan, dtype=float)

        valid_samples = (
            (n_valid_pixels > 0) & np.isfinite(noise_sigma) & (noise_sigma > 0)
        )

        mean_residual[valid_samples] = (
            np.sum(
                np.where(valid_pixels[valid_samples], residuals[valid_samples], 0.0),
                axis=(1, 2),
            )
            / n_valid_pixels[valid_samples]
        )

        chi2_mad[valid_samples] = np.sum(
            np.where(
                valid_pixels[valid_samples],
                (residuals[valid_samples] - mean_residual[valid_samples, None, None])
                ** 2
                / noise_sigma[valid_samples, None, None] ** 2,
                0.0,
            ),
            axis=(1, 2),
        )

        return chi2_mad

    @staticmethod
    def _compute_chi2_mad_metrics(
        residuals: np.ndarray,
        valid_pixels: np.ndarray,
        noise_sigma: np.ndarray,
        n_valid_pixels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-sample chi-square and reduced chi-square using MAD noise estimates.

        Notes
        -----
        Reduced chi-square is calculated using ``n_valid_pixels - 1`` degrees
        of freedom. Samples with fewer than two valid pixels receive NaN for
        reduced chi-square.
        """
        chi2_mad = GoodnessOfFitMetric._compute_chi2_mad(
            residuals=residuals,
            valid_pixels=valid_pixels,
            noise_sigma=noise_sigma,
            n_valid_pixels=n_valid_pixels,
        )

        reduced_chi2_mad = np.full(chi2_mad.shape, np.nan, dtype=float)

        valid_samples = np.isfinite(chi2_mad) & (n_valid_pixels > 1)

        reduced_chi2_mad[valid_samples] = chi2_mad[valid_samples] / (
            n_valid_pixels[valid_samples] - 1.0
        )

        return chi2_mad, reduced_chi2_mad

    @staticmethod
    def _compute_weighted_mean_residual(
        residuals: np.ndarray,
        usable_pixels: np.ndarray,
        icovs: np.ndarray,
    ) -> np.ndarray:
        """Compute the inverse-variance-weighted mean residual per sample.

        Parameters
        ----------
        residuals : np.ndarray
            Residual images with shape (n_samples, height, width).

        usable_pixels : np.ndarray
            Boolean mask identifying pixels included in the calculation,
            with True indicating a usable pixel.

        icovs : np.ndarray
            Inverse variance maps with the same shape as ``residuals``.

        Returns
        -------
        np.ndarray
            Weighted mean residual per sample. Samples with no usable pixels
            receive NaN.

        Notes
        -----
        Only usable pixels contribute to the weighted mean. Pixels with
        non-finite or non-positive inverse variances must be excluded
        through ``usable_pixels``.
        """
        weighted_residuals = np.sum(
            np.where(usable_pixels, residuals * icovs, 0.0),
            axis=(1, 2),
        )

        sum_icovs = np.sum(
            np.where(usable_pixels, icovs, 0.0),
            axis=(1, 2),
        )

        mean_residual = np.full(residuals.shape[0], np.nan, dtype=float)
        valid_samples = sum_icovs > 0
        mean_residual[valid_samples] = (
            weighted_residuals[valid_samples] / sum_icovs[valid_samples]
        )

        return mean_residual

    @staticmethod
    def _compute_chi2_icov(
        residuals: np.ndarray,
        icovs: np.ndarray,
        valid_pixels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute inverse-variance-weighted chi-square over usable pixels.

        Parameters
        ----------
        residuals : np.ndarray
            Residual images with shape (n_samples, height, width).

        icovs : np.ndarray
            Inverse variance maps with the same shape as ``residuals``.

        valid_pixels : np.ndarray
            Boolean mask identifying valid pixels, with True indicating a
            valid pixel.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Chi-square values and number of usable pixels per sample.
            Samples with no usable pixels have a chi-square of zero and a
            usable-pixel count of zero.

        Raises
        ------
        ValueError
            If residuals and valid_pixels have incompatible dimensions or
            shapes, or if icovs does not have the same shape as residuals.

        Notes
        -----
        The weighted mean residual is subtracted from each sample's residuals
        before computing chi-square. Only pixels with a valid mask, finite
        inverse variance, and strictly positive inverse variance are included.
        """
        GoodnessOfFitMetric._validate_residual_inputs(
            residuals=residuals,
            valid_pixels=valid_pixels,
        )

        if icovs.shape != residuals.shape:
            raise ValueError("'icovs' must have the same shape as 'residuals'.")

        usable_pixels = valid_pixels & np.isfinite(icovs) & (icovs > 0)

        mean_residual = GoodnessOfFitMetric._compute_weighted_mean_residual(
            residuals=residuals, usable_pixels=usable_pixels, icovs=icovs
        )

        chi2_icov = np.sum(
            np.where(
                usable_pixels,
                (residuals - mean_residual[:, None, None]) ** 2 * icovs,
                0.0,
            ),
            axis=(1, 2),
        )

        n_valid_icov_pixels = np.sum(usable_pixels, axis=(1, 2))

        return chi2_icov, n_valid_icov_pixels

    @staticmethod
    def _compute_chi2_icov_metrics(
        residuals: np.ndarray,
        icovs: np.ndarray,
        valid_pixels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-sample chi-square and reduced chi-square using inverse variances.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Chi-square values, reduced chi-square values, and number of
            usable pixels per sample.

        Notes
        -----
        Reduced chi-square is calculated using ``n_valid_icov_pixels - 1``
        degrees of freedom. Samples with fewer than two usable pixels receive
        NaN for reduced chi-square.
        """
        chi2_icov, n_valid_icov_pixels = GoodnessOfFitMetric._compute_chi2_icov(
            residuals=residuals,
            icovs=icovs,
            valid_pixels=valid_pixels,
        )
        reduced_chi2_icov = np.full(chi2_icov.shape, np.nan, dtype=float)
        valid_samples = n_valid_icov_pixels > 1

        reduced_chi2_icov[valid_samples] = chi2_icov[valid_samples] / (
            n_valid_icov_pixels[valid_samples] - 1
        )

        return chi2_icov, reduced_chi2_icov, n_valid_icov_pixels
