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
from .operators import COMPARISON_OPERATORS, validate_comparison_operator

import numpy as np
from numpy.typing import NDArray
from typing import Any


class ThresholdRejectionPolicy(RejectionPolicy):
    """Reject dataset samples based on configurable metric threshold.

    Attributes
    ----------
    name : str
        Policy identifier used by the rejection policy registry.

    value : float
        Threshold applied by the rejection policy.

    operator : str
        Comparison operator applied as ``metric operator threshold_value``. Observations satisfying the comparison are retained (True in the validity mask); observations that do not are rejected (False).

    """

    name = "threshold"

    @classmethod
    def validate_config(cls, params: dict[str, Any]) -> None:
        """Validate threshold rejection policy configuration."""
        operator = params.get("operator")
        validate_comparison_operator(operator=operator)

        value = params.get("value")

        if value is None:
            raise ValueError("Threshold rejection policy requires a 'value' parameter.")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Threshold value must be a float or an int.")

        if not np.isfinite(value):
            raise ValueError("Threshold value must be finite.")

    def __init__(self, value: float, operator: str):
        self.validate_config({"value": value, "operator": operator})

        self.value = value
        self.operator = operator
        self._compare = COMPARISON_OPERATORS[operator]

    def apply(self, metric: np.ndarray) -> NDArray[np.bool_]:
        """Apply threshold-based rejection to metric values."""
        return self._compare(metric, self.value)
