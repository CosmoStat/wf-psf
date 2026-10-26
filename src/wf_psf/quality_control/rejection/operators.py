"""Comparison operators available to quality control rejection policies.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""

from operator import ge, gt, le, lt

COMPARISON_OPERATORS = {
    "<": lt,
    "<=": le,
    ">": gt,
    ">=": ge,
}
