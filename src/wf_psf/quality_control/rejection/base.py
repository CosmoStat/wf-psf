"""Base interfaces for rejection policies.

Defines the abstract interface implemented by rejection policies.

Rejection policies convert quality metrics into boolean validity masks.
These masks are used by the quality control pipeline to identify valid
dataset samples for downstream processing, such as model training,
evaluation, or inference.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from abc import ABC, abstractmethod
import numpy as np


class RejectionPolicy(ABC):
    """Abstract interface for sample rejection policy implementations.

    Attributes
    ----------
    name : str
        Unique identifier for the rejection policy implementation. Used by
        the RejectionPolicyRegistry to register and retrieve rejection
        policy classes.

    Methods
    -------
    apply(metric)
        Apply the rejection policy to the supplied quality metric and
        return a boolean validity mask.

    """

    name: str

    @abstractmethod
    def apply(self, metric: np.ndarray) -> np.ndarray:
        """Apply the rejection policy to metric values.

        Returns
        -------
        np.ndarray
            Boolean validity mask with one entry per dataset sample. ``True``
            indicates that the sample passes the rejection policy and should
            be retained; ``False`` indicates that it should be rejected.
        """
