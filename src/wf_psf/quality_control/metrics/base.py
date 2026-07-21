"""Base interfaces for quality metrics.

Defines the abstract interface implemented by all quality metrics used
by the quality control framework.

Quality metrics compute numerical measures describing dataset quality.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
from abc import ABC, abstractmethod

class QualityMetric(ABC):
    """Abstract interface for quality metric implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of quality metric."""
        ...

    @abstractmethod
    def compute(self, dataset):
        """Compute one metric value for each dataset sample."""
       