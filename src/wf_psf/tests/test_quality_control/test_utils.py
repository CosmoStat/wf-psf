"""Test utilities for quality control tests.

This module contains utilities and classes for setting up tests in the
quality control package.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np


class PSFDataset:
    """Simple dataset of numpy arrays."""

    def __init__(self, n_src: int = 10, n_pix: int = 32):
        self.n_src = n_src
        self.n_pix = n_pix
        self.positions = np.arange(n_src * 2).reshape(n_src, 2)
        self.masks = np.zeros((n_src, n_pix, n_pix), dtype=bool)
