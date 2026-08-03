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


class ThresholdRejectionPolicy(RejectionPolicy):
    """Reject dataset samples according to configurable metric thresholds."""

    name = "threshold"

    def apply(self, metric):
        """Apply threshold policy."""
        raise NotImplementedError
