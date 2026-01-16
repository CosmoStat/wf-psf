"""Read Config.

A module which defines methods to
read configuration files.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import yaml
from yaml.scanner import ScannerError
from yaml.parser import ParserError
from types import SimpleNamespace
import logging

logger = logging.getLogger(__name__)


class RecursiveNamespace(SimpleNamespace):
    """RecursiveNamespace.

    A child class of the type SimpleNamespace to
    create nested namespaces (objects).

    Parameters
    ----------
    **kwargs
        Extra keyword arguments used to build
        a nested namespace.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, RecursiveNamespace(**val))
            elif isinstance(val, list):
                setattr(self, key, list(map(self.map_entry, val)))

    @staticmethod
    def map_entry(entry):
        """Map Entry.

        A function to map a dictionary to a
        RecursiveNamespace object.

        Parameters
        ----------
        entry: type

        Returns
        -------
        RecursiveNamespace
            RecursiveNamespace object if entry type is a dictionary
        entry: type
            Original type of entry if type is not a dictionary
        """
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry


def read_yaml(conf_file):
    """Read Yaml.

    A function to read a YAML file.

    Parameters
    ----------
    conf_file: str
        Name of configuration file

    Returns
    -------
    config: dict
        A dictionary containing configuration parameters.
    """
    with open(conf_file) as f:
        config = yaml.safe_load(f)

    return config


def read_conf(conf_file):
    """Read Conf.

    A function to read a yaml configuration file, recursively.

    Parameters
    ----------
    conf_file: str
        Name of configuration file

    Returns
    -------
    RecursiveNamespace
        Recursive Namespace object

    """
    logger.info(f"Loading...{conf_file}")
    with open(conf_file) as f:
        try:
            my_conf = yaml.safe_load(f)
        except (ParserError, ScannerError, TypeError):
            logger.exception(
                f"There is a syntax problem with your config file. Please check {conf_file}."
            )
            exit()

        if my_conf is None:
            raise TypeError(f"Config file {conf_file} is empty...Stopping Program.")
            exit()

        try:
            return RecursiveNamespace(**my_conf)
        except TypeError as e:
            logger.exception(f"Check your config file for errors. Error Msg: {e}.")
            exit()


def read_stream(conf_file):
    """Read Stream.

    A generator to read multiple docs in a yaml config.

    Parameters
    ----------
    conf_file : str
        Name of configuration file

    Yields
    ------
    dict
        A dictionary containing all config files.

    """
    stream = open(conf_file)
    docs = yaml.load_all(stream, yaml.FullLoader)

    for doc in docs:  # noqa: UP028
        yield doc  
