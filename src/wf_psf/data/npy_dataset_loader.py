"""NpyDatasetLoader.

Loads datasets stored as serialized NumPy `.npy` files.

This loader is format-based and can be used for both simulated
and real datasets provided they are saved in the expected `.npy`
dictionary format.

Author: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import os
import numpy as np


class NpyDatasetLoader:
    """
    Loads datasets stored as NumPy `.npy` files.

    Parameters
    ----------
    data_params : RecursiveNamespace
        Configuration object containing dataset parameters
        (e.g., data directory and file name).

    Attributes
    ----------
    data_params : RecursiveNamespace
        Configuration parameters for data access and structure.
    dataset : dict
        Loaded dataset including keys such as 'positions', 'sources',
        'seds', etc.
    """

    def __init__(self, data_params):
        self.data_params = data_params
        self.dataset = None

    def load(self):
        """Load dataset from disk."""
        path = os.path.join(self.data_params.data_dir, self.data_params.file)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self.dataset = np.load(path, allow_pickle=True)[()]
