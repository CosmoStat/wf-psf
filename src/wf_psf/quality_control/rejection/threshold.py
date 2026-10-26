"""Threshold-based rejection policies.

Provides implementations of rejection policies that convert quality
metrics into boolean validity masks using configurable threshold
criteria.

Additional rejection policies may be added in the future without
modifying the quality control pipeline.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from .base import RejectionPolicy
from .operators import COMPARISON_OPERATORS

import numpy as np
from numpy.typing import NDArray


class ThresholdRejectionPolicy(RejectionPolicy):
    """Reject dataset samples based on configurable metric threshold.

    Attributes
    ----------
    name : str
        Policy identifier used by the rejection policy registry.

    value : float
        Threshold applied by the rejection policy.

    operator : str
        Operator used by the rejection policy to evaluate metric against the threshold value.

    """

    name = "threshold"

    def __init__(self, value: float, operator: str = ">"):
        if operator not in COMPARISON_OPERATORS:
            raise ValueError(f"Unsupported threshold operator '{operator}'.")

        self.value = value
        self.operator = operator
        self._compare = COMPARISON_OPERATORS[operator]

    def apply(self, metric: np.ndarray) -> NDArray[np.bool_]:
        """Apply threshold-based rejection to metric values."""
        return self._compare(metric, self.value)
