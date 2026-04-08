"""NpyDatasetLoader.

Loads datasets stored as serialized NumPy `.npy` files.

This loader is format-based and can be used for both simulated
and real datasets provided they are saved in the expected `.npy`
dictionary format.

Author: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import numpy as np
from pathlib import Path


class NpyDatasetLoader:
    """
    Loads datasets stored as NumPy `.npy` files.

    The file is expected to contain a pickled Python dictionary
    mapping dataset field names (e.g. 'positions', 'seds')
    to NumPy arrays.

    Required dataset fields
    -----------------------
    positions : ndarray
        Star positions with shape (n_samples, 2).

    seds : ndarray
        Spectral energy distributions with shape (n_samples, n_bins, 2).

    target_field : ndarray
        Array containing the target images (e.g. stars). The actual key
        name may vary and is specified by the ``target_field`` parameter
        in the data configuration file.

    Optional dataset fields
    -----------------------
    masks : ndarray, optional
        Pixel masks associated with target images.
    zernike_prior : ndarray, optional
        Zernike coefficient prior information.

    Notes
    -----
    This loader performs no validation of dataset fields. Field
    validation and canonical key handling are managed later by the
    data adapter pipeline.

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
        path = Path(self.data_params.data_dir) / self.data_params.file

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        self.dataset = np.load(path, allow_pickle=True)[()]
