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
def test_validate_config_valid_operator(operator):
    ThresholdRejectionPolicy.validate_config({"value": 0.25, "operator": operator})


@pytest.mark.parametrize("value", [-10, 0, 10.0])
def test_validate_config_valid_value(value):
    ThresholdRejectionPolicy.validate_config({"value": value, "operator": "<"})


def test_validate_config_missing_operator():
    with pytest.raises(
        ValueError,
        match="A comparison operator must be specified.",
    ):
        ThresholdRejectionPolicy.validate_config({"value": 0.25})


def test_validate_config_missing_value():
    with pytest.raises(
        ValueError,
        match="Threshold rejection policy requires a 'value' parameter.",
    ):
        ThresholdRejectionPolicy.validate_config({"operator": "<"})


def test_validate_config_none_value():
    with pytest.raises(
        ValueError,
        match="Threshold rejection policy requires a 'value' parameter.",
    ):
        ThresholdRejectionPolicy.validate_config({"operator": "<", "value": None})


@pytest.mark.parametrize(
    "value, error, err_msg",
    [
        (np.nan, ValueError, "Threshold value must be finite."),
        (np.inf, ValueError, "Threshold value must be finite."),
        (-np.inf, ValueError, "Threshold value must be finite."),
        (True, TypeError, "Threshold value must be a float or an int."),
        ("0.25", TypeError, "Threshold value must be a float or an int."),
    ],
)
def test_validate_config_invalid_value(value, error, err_msg):
    with pytest.raises(error, match=err_msg):
        ThresholdRejectionPolicy.validate_config({"operator": "<", "value": value})


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
