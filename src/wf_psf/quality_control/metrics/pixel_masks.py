"""Mask obscuration quality metric.

Defines a quality metric implementation for assessing the impact of
masked pixels on dataset samples.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from .base import QualityMetric
from typing import ClassVar


class PixelMaskMetric(QualityMetric):
    """Evaluate pixel-mask metrics for each dataset sample."""

    name = "pixel_mask"
    diagnostics: ClassVar[frozenset[str]] = frozenset(
        {
            "total_masked_pixels",
            "total_masked_fraction",
            "aperture_masked_pixels",
            "aperture_masked_fraction",
        }
    )

    def compute(self, dataset):
        """Compute pixel-mask metrics for each dataset sample."""
        pass
