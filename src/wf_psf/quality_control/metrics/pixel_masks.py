"""Mask obscuration quality metric.

Defines a quality metric implementation for assessing the impact of
masked pixels on dataset samples.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from wf_psf.quality_control.context import QualityControlContext

from .base import QualityMetric
import numpy as np


class PixelMaskMetric(QualityMetric):
    """Evaluate pixel-mask metrics for each dataset sample."""

    name = "pixel_mask"

    def compute(self, context: QualityControlContext) -> dict[str, np.ndarray]:
        """Compute pixel-mask metrics for each dataset sample.

        Parameters
        ----------
        context : QualityControlContext
            Quality control context containing the dataset and resources required to evaluate the metric.

        Returns
        -------
        dict[str, np.ndarray]
            Masked-pixel diagnostics for each image, consisting of the number and fraction of masked pixels over the entire image and within the configured aperture.

        """
        masks = self._get_masks(context)

        aperture_masked_pixels, aperture_masked_fraction = (
            self._compute_aperture_masked_pixel_stats(context)
        )

        return {
            "total_masked_pixels": np.sum(masks, axis=(1, 2)),
            "total_masked_fraction": np.mean(masks, axis=(1, 2)),
            "aperture_masked_pixels": aperture_masked_pixels,
            "aperture_masked_fraction": aperture_masked_fraction,
        }

    def _get_masks(self, context: QualityControlContext) -> np.ndarray:
        """Return the dataset pixel masks as boolean arrays."""
        return context.dataset.masks.astype(bool)

    def _get_centres(self, context: QualityControlContext):
        """Return aperture centres according to the configured centre definition.

        Parameters
        ----------
        context : QualityControlContext
            Quality control context containing the dataset from which the aperture centres are obtained.

        Returns
        -------
        np.ndarray
            Aperture centre coordinates with shape ``(n_samples, 2)``.  Aperture centres are expressed in pixel coordinates, where integer coordinates correspond to pixel centres.

        Raises
        ------
        ValueError
            If the configured aperture centre is not recognised.


        """
        centre = self.params["aperture"]["centre"]

        if centre == "centroid":
            return context.dataset.positions

        if centre == "stamp_centre":
            ny, nx = context.dataset.masks.shape[-2:]
            return np.full(
                (context.dataset.masks.shape[0], 2),
                ((nx - 1) / 2, (ny - 1) / 2),
            )

        raise ValueError(f"Unknown aperture centre: '{centre}'.")

    def _get_aperture_radius(self) -> float:
        """Return the configured aperture radius in pixels.

        Raises
        ------
        ValueError
            If the aperture unit is unknown or the aperture radius is non-finite or non-positive.

        """
        aperture = self.params["aperture"]

        if aperture["unit"] != "pixel":
            raise ValueError(f"Unknown aperture unit: '{aperture['unit']}'.")

        radius = aperture["radius"]

        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(
                f"Aperture radius must be finite and positive, got '{radius}'."
            )

        return radius

    def _get_aperture_mask(self, context: QualityControlContext) -> np.ndarray:
        """Return circular aperture masks centred on the configured positions.

        Parameters
        ----------
        context : QualityControlContext
            Quality control context containing the dataset used to determine the aperture geometry.

        Returns
        -------
        np.ndarray
            Boolean aperture mask for each dataset sample, with shape ``(n_samples, ny, nx)`` where `ny` and `nx` are the number of rows and columns, respectively.

        Notes
        -----
        Coordinates are represented as (x, y), with x corresponding to columns and y corresponding to rows.
        """
        mask = context.dataset.masks
        centres = self._get_centres(context)
        radius = self._get_aperture_radius()

        ny, nx = mask.shape[-2:]

        y, x = np.ogrid[:ny, :nx]

        aperture = (
            np.hypot(
                y[None, :, :] - centres[:, 1, None, None],
                x[None, :, :] - centres[:, 0, None, None],
            )
            <= radius
        )

        return aperture

    def _compute_aperture_masked_pixel_stats(
        self, context: QualityControlContext
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute masked-pixel diagnostics within the configured aperture.

        Parameters
        ----------
        context : QualityControlContext
            Quality control context containing the dataset used to compute the diagnostics.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Number of masked pixels within the aperture and fraction of aperture pixels that are masked, respectively. Both arrays have shape ``(n_samples,)``.

        Raises
        ------
        ValueError
            If the aperture contains no pixels for one or more samples, which would result in division by zero when computing the masked-pixel fraction.

        """
        masks = self._get_masks(context)
        aperture = self._get_aperture_mask(context=context)

        n_aperture_masked_pixels = np.sum(masks & aperture, axis=(1, 2))
        n_aperture_pixels = np.sum(aperture, axis=(1, 2))

        if np.any(n_aperture_pixels == 0):
            raise ValueError("Aperture contains no pixels for one or more samples.")

        aperture_masked_fraction = n_aperture_masked_pixels / n_aperture_pixels

        return n_aperture_masked_pixels, aperture_masked_fraction
