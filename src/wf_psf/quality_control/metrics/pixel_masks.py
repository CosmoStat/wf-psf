"""Mask obscuration quality metric.

Defines a quality metric implementation for assessing the impact of
masked pixels on dataset samples.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from .base import QualityMetric


class PixelMaskMetric(QualityMetric):
    """Evaluate pixel-mask metrics for each dataset sample."""

    name = "pixel_mask"

    def compute(self, dataset):
        """Compute pixel-mask metrics for each dataset sample."""
        pass
