"""Configs_Handler Registry.

Provides runtime dispatch for config handlers based on config file names.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from abc import ABC, abstractmethod
import logging
import re
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


logger = logging.getLogger(__name__)

CONFIG_CLASS = {}


class ConfigHandler(ABC):
    """Abstract base class for all config handlers."""

    ids: tuple[str, ...]

    @abstractmethod
    def run(self):
        """Execute the configured task."""
        pass


def register_configclass(config_class):
    """Register Config Class.

    A wrapper function to register all config classes
    in a dictionary.

    Parameters
    ----------
    config_class: type
        Config Class

    Returns
    -------
    config_class: type
        Config class

    """
    for id in config_class.ids:
        CONFIG_CLASS[id] = config_class

    return config_class


def set_run_config(config_name):
    """Set Run Configuration Class.

    A function to retrieve the appropriate configuration
    class based on the provided config name.

    Parameters
    ----------
    config_name: str
        Name of config

    Returns
    -------
    config_class: class
        Name of config class

    """
    try:
        config_id = [id for id in CONFIG_CLASS.keys() if re.search(id, config_name)][0]
        config_class = CONFIG_CLASS[config_id]
    except KeyError:
        logger.exception("Invalid config name. Check your config settings.")
        exit()

    return config_class


def get_run_config(run_config_name, *config_params):
    """Get Run Configuration Instance.

    A function to retrieve an instance of
    the appropriate configuration class for
    a WF-PSF run.

    Parameters
    ----------
    run_config_name: str
        Name of the run configuraton
    *config_params: str
        Run configuration parameters used for class instantiation.

    Returns
    -------
    config_class: object
        A class instance of the selected configuration class.

    """
    config_class = set_run_config(run_config_name)

    return config_class(*config_params)


class ConfigParameterError(Exception):
    """Custom Config Parameter Error exception class for specific error scenarios."""

    def __init__(self, message="An error with your config settings occurred."):
        self.message = message
        super().__init__(self.message)


# Lazy imports for backward compatibility (avoids circular imports)
def __getattr__(name):
    """
    Lazy import handler for backward compatibility.

    Allows old code like:
        from wf_psf.utils.configs_handler import TrainingConfigHandler

    to continue working without circular imports.
    """
    _HANDLER_MAP = {
        "TrainingConfigHandler": "wf_psf.training.training_config_handler",
        "MetricsConfigHandler": "wf_psf.metrics.metrics_config_handler",
        "PlottingConfigHandler": "wf_psf.plotting.plotting_config_handler",
        "DataConfigHandler": "wf_psf.data.data_config_handler",
    }

    if name in _HANDLER_MAP:
        import importlib

        module = importlib.import_module(_HANDLER_MAP[name])
        return getattr(module, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    from wf_psf.training.training_config_handler import TrainingConfigHandler
    from wf_psf.metrics.metrics_config_handler import MetricsConfigHandler
    from wf_psf.plotting.plotting_config_handler import PlottingConfigHandler
    from wf_psf.data.data_config_handler import DataConfigHandler


__all__ = [
    "ConfigHandler",
    "register_configclass",
    "set_run_config",
    "get_run_config",
    "ConfigParameterError",
    "CONFIG_CLASS",
    # Handlers available via lazy import
    "TrainingConfigHandler",
    "MetricsConfigHandler",
    "PlottingConfigHandler",
    "DataConfigHandler",
]
