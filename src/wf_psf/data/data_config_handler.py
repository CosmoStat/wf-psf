"""DataConfigHandler.

A module which provides a class to manage the parameters of the data config file.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from wf_psf.utils.configs_handler import ConfigHandler
from wf_psf.utils.read_config import read_conf
import logging


logger = logging.getLogger(__name__)


class DataConfigHandler(ConfigHandler):
    """DataConfigHandler.

    A class to handle data configuration
    parameters.

    Parameters
    ----------
    data_conf : str
        Path of the data configuration file
    """

    ids = ("data_conf",)

    DEFAULTS = {
        "train_fraction": 0.8,
        "seed": 42,
        "canonical_keys": ["sources", "positions", "masks", "seds"],
    }

    def __init__(self, data_conf):
        try:
            self.params = read_conf(data_conf)
        except (FileNotFoundError, TypeError) as e:
            logger.exception(e)
            exit()
        # Normalize parameters
        self.run()

    def run(self):
        """Run DataConfigHandler.

        A function to run the data configuration handler.

        """
        """Normalize and validate data configuration."""

        params = self.params

        # Apply defaults
        for key, value in self.DEFAULTS.items():
            if getattr(params, key, None) is None:
                setattr(params, key, value)

        # Validate train_fraction
        if not 0 < params.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")

        # Ensure canonical_keys is a list
        if not isinstance(params.canonical_keys, list):
            raise TypeError("canonical_keys must be a list")

        self.params = params
