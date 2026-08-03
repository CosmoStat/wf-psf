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
import numpy as np


class ThresholdRejectionPolicy(RejectionPolicy):
    """Reject dataset samples based on configurable metric thresholds."""

    name = "threshold"

    def apply(self, metric: np.ndarray) -> np.ndarray:
        """Apply threshold-based rejection to metric values."""
        raise NotImplementedError
