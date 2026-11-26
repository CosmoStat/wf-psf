"""Comparison operators available to quality control rejection policies.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""
from __future__ import annotations

from operator import ge, gt, le, lt
from types import MappingProxyType

COMPARISON_OPERATORS = MappingProxyType(
    {
        "<": lt,
        "<=": le,
        ">": gt,
        ">=": ge,
    }
)


def validate_comparison_operator(operator: str | None) -> None:
    """Validate a comparison operator.

    Raises
    ------
    ValueError
        If the operator is unsupported.
    """
    if operator is None:
        raise ValueError("A comparison operator must be specified.")

    if not isinstance(operator, str):
        raise ValueError("Comparison operator must be a string.")

    if operator not in COMPARISON_OPERATORS:
        supported = ", ".join(COMPARISON_OPERATORS)
        raise ValueError(
            f"Unsupported operator '{operator}'. Supported operators are: {supported}."
        )
