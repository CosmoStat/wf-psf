"""Base interfaces for quality metrics.

Defines the abstract interface implemented by all quality metrics used
by the quality control framework.

Quality metrics compute numerical measures describing the suitability
of individual dataset samples for downstream processing, such as model
training, evaluation, or inference.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any


class QualityMetric(ABC):
    """Abstract interface for quality metric implementations.

    Attributes
    ----------
    name : str
        Unique identifier for the metric implementation. Used by
        the MetricsRegistry to register and retrieve metric classes.

    params : dict[str, Any]
        Parameter set for configuring a specific metric.

    Methods
    -------
    compute(dataset)
        Compute the quality metric for the supplied dataset.

    """

    name: str

    def __init__(self, params: dict[str, Any]):
        self.params = params

    @abstractmethod
    def compute(self, dataset: Any) -> dict[str, np.ndarray]:
        """Compute the quality metric for the supplied dataset.

        Parameters
        ----------
        dataset
            Dataset containing the samples to be evaluated. A dataset may
            contain multiple postage stamps/images and their associated
            metadata or auxiliary data, depending on the requirements of
            the metric.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of diagnostic names to arrays containing one value per
            dataset sample. All returned arrays must be aligned with the dataset
            samples.
        """
