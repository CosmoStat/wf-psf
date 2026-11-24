"""UNIT TESTS FOR PACKAGE MODULE: Quality Control Rejection Policy Operators.

This module contains unit tests for the rejection policy operators module.

:Author:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import pytest

from wf_psf.quality_control.rejection.operators import (
    COMPARISON_OPERATORS,
    validate_comparison_operator,
)


@pytest.mark.parametrize("operator", list(COMPARISON_OPERATORS))
def test_validate_comparison_operator_supported(operator):
    validate_comparison_operator(operator=operator)


def test_validate_comparison_operator_unsupported():
    operator = "foo"

    with pytest.raises(ValueError, match="Unsupported operator"):
        validate_comparison_operator(operator=operator)


def test_validate_comparison_operator_missing():
    with pytest.raises(ValueError, match="A comparison operator must be specified"):
        validate_comparison_operator(operator=None)


@pytest.mark.parametrize("operator", [1, 1.5, True, [], {}])
def test_validate_comparison_operator_invalid_type(operator):
    with pytest.raises(ValueError, match="Comparison operator must be a string"):
        validate_comparison_operator(operator=operator)
