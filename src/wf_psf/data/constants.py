"""
Constants for dataset handling and preprocessing.

This module defines shared default values used across the data pipeline,
including canonical internal dataset field names and default parameters 
used throughout wf_psf for dataset processing operations.

External datasets may use arbitrary field names which are mapped
to these canonical identifiers through the data configuration
and adapter layers.

Author(s): Jennifer Pollack <jennifer.pollack@cea.fr>
"""

CANONICAL_DATASET_KEYS = (
    "sources",
    "positions",
    "seds",
)  # canonical dataset fields
OPTIONAL_KEYS = ("masks", "zernike_prior")
DATASET_INDEX_KEY = "positions"
DEFAULT_TRAIN_FRACTION = 0.8  # default train/test split ratio
DEFAULT_SEED = 42  # default RNG seed for reproducible splits

## Handler keys
# Key for SED processing handler
SED_DOMAIN = "seds"