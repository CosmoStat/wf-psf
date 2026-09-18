"""Goodness of fit quality metric.

Defines the goodness-of-fit quality class with methods for computing
various statistics (e.g. Chi-square) to quantify model agreement with
data.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from wf_psf.quality_control.context import QualityControlContext

from .base import QualityMetric
import numpy as np


class GoodnessOfFitMetric(QualityMetric):
    """Compute a goodness-of-fit metric (e.g. reduced chi square) for each dataset sample."""

    name = "goodness_of_fit"

    def compute(self, context: QualityControlContext) -> dict[str, np.ndarray]:
        """Compute reduced chi-square values for each dataset sample."""
        ...
