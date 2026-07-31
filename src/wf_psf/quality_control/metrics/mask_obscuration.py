"""Mask obscuration quality metric.

Defines the mask obscuration metric class with techniques for quantifying
the number of pixels associated to a source that have been masked.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from .base import QualityMetric


class MaskObscurationMetric(QualityMetric):
    """Mask Obscuration metric class.

    Attributes
    ----------
    name : str

    Functions
    ---------
    compute
    """

    name = "mask_obscuration"

    def compute(self, dataset):
        """Compute metric."""
        pass
