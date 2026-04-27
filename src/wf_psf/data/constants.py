"""
Constants for dataset handling and preprocessing.

This module defines shared default values used across the data pipeline,
including canonical dataset field names and default parameters for
dataset processing operations such as train/test splitting.

These constants provide a single source of truth for components such as
``DataAdapter`` and ``DataConfigHandler``, ensuring consistent dataset handling
throughout the library.

Author(s): Jennifer Pollack <jennifer.pollack@cea.fr>
"""

DEFAULT_CANONICAL_KEYS = [
    "sources",
    "positions",
    "seds",
]  # canonical dataset fields
OPTIONAL_KEYS = ["masks", "zernike_prior"]
DEFAULT_TRAIN_FRACTION = 0.8  # default train/test split ratio
DEFAULT_SEED = 42  # default RNG seed for reproducible splits
