"""SimulationDataLoader.

Loads simulation data from .npy files on disk for training/testing.

Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import os


class SimulationDataLoader:
    """
    Loads simulation data from .npy files on disk.

    Use this for training/testing with pre-saved simulation datasets.

    Parameters
    ----------
    data_params : RecursiveNamespace
        Configuration object containing dataset parameters (e.g., file paths, preprocessing flags).
    dataset : dict or list, optional
        If provided, uses this pre-loaded dataset instead of triggering automatic loading.

    Attributes
    ----------
    data_params : RecursiveNamespace
        Configuration parameters for data access and structure.
    dataset : dict
        Loaded dataset including keys such as 'positions', 'stars', 'noisy_stars', or similar.

    """

    def __init__(self, data_params):
        self.data_params = data_params
        self.dataset = None

    def load(self):
        """Load .npy file."""
        self._load_from_disk()

    def _load_from_disk(self):
        """Load .npy file."""
        self.dataset = np.load(
            os.path.join(self.data_params.data_dir, self.data_params.file),
            allow_pickle=True,
        )[()]
