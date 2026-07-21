"""Base interfaces for rejection policies.

Defines the abstract interface implemented by rejection policies.

Rejection policies convert quality metrics into boolean validity masks.
This separation allows metric computation and rejection decisions to
evolve independently.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from abc import ABC, abstractmethod

class RejectionPolicy(ABC):
    """Abstract interface for quality-based sample rejection."""

    @abstractmethod
    def apply(self, metric):
        """Return a boolean validity mask from a quality metric."""