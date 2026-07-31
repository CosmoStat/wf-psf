"""Base interfaces for quality metrics.

Defines the abstract interface implemented by all quality metrics used
by the quality control framework.

Quality metrics compute numerical measures describing the suitability
of individual dataset samples for downstream processing, such as model
training, evaluation, or inference.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from abc import ABC, abstractmethod


class QualityMetric(ABC):
    """Abstract interface for quality metric implementations."""

    name: str

    @abstractmethod
    def compute(self, dataset):
        """Compute one metric value for each dataset sample."""
