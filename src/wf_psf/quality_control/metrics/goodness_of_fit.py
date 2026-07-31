"""Goodness of fit quality metric.

Defines the goodness-of-fit quality class with methods for computing
various statistics (e.g. Chi-square) to quantify model agreement with
data.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from .base import QualityMetric


class GoodnessOfFitMetric(QualityMetric):
    """Goodness-of-fit metric class.

    Attributes
    ----------
    name : str

    Functions
    ---------
    compute
    """

    name = "goodness_of_fit"

    def compute(self, dataset):
        """Compute metric."""
        pass
