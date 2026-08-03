"""Mask obscuration quality metric.

Defines a quality metric implementation for assessing the impact of
masked pixels on dataset samples.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from .base import QualityMetric


class MaskObscurationMetric(QualityMetric):
    """Assess masked pixel obscuration for each dataset sample."""

    name = "mask_obscuration"

    def compute(self, dataset):
        """Compute the mask obscuration metric for each dataset sample."""
        pass
