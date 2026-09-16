"""Mask obscuration quality metric.

Defines a quality metric implementation for assessing the impact of
masked pixels on dataset samples.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from .base import QualityMetric
import numpy as np
from typing import Any


class PixelMaskMetric(QualityMetric):
    """Evaluate pixel-mask metrics for each dataset sample."""

    name = "pixel_mask"

    def compute(self, dataset: Any) -> dict[str, np.ndarray]:
        """Compute pixel-mask metrics for each dataset sample."""
        ...
