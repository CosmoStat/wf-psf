"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Rejection Threshold.

This module contains unit tests for the ThresholdRejectionPolicy class.

:Author:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import pytest
import numpy as np

from wf_psf.quality_control.rejection.operators import COMPARISON_OPERATORS


from wf_psf.quality_control.rejection.threshold import ThresholdRejectionPolicy


@pytest.mark.parametrize("operator", list(COMPARISON_OPERATORS))
def test_threshold_rejection_policy_constructor_supported_operator(operator):
    threshold_policy = ThresholdRejectionPolicy(value=0.2, operator=operator)
    assert threshold_policy.value == 0.2
    assert threshold_policy.operator == operator
    assert threshold_policy._compare == COMPARISON_OPERATORS[operator]


def test_threshold_rejection_policy_constructor_unsupported_operator():
    operator = "foo"
    with pytest.raises(
        ValueError, match=f"Unsupported threshold operator '{operator}'."
    ):
        _ = ThresholdRejectionPolicy(value=0.2, operator=operator)


@pytest.mark.parametrize(
    ("operator, expected"),
    [
        ("<", [True, False, False]),
        ("<=", [True, True, False]),
        (">", [False, False, True]),
        (">=", [False, True, True]),
    ],
)
def test_threshold_rejection_policy_apply_valid(operator, expected):
    metric = np.array([0.1, 0.2, 0.3])
    value = 0.2
    threshold_policy = ThresholdRejectionPolicy(value=value, operator=operator)

    result = threshold_policy.apply(metric)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, expected)
