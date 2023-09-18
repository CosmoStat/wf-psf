"""Read Config.

A module which defines methods to
read configuration files.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import yaml
import pprint
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
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
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
    logger.info(conf_file)
    with open(conf_file, "r") as f:
        try:
            my_conf = yaml.safe_load(f)
        except yaml.scanner.ScannerError as e:
            logger.exception("Improper syntax in yaml file.")
            exit()
        except TypeError:
            logger.exception("There is a problem with your config file. Please check.")
    
        if my_conf == None:
            logger.info("Config file is empty...Stopping Program.")
            exit()

        try:
            return RecursiveNamespace(**my_conf)
        except TypeError:
            logger.exception("Check your config file for Syntax error key:value pair mapping expected.") 
            exit()
          
        


def read_stream(conf_file):
    """Read Stream.

    A generator to read multiple docs in a yaml config.

    Parameters
    ----------
    conf_file
        Name of configuration file

    Yields
    ------
    dict
        A dictionary containing all config files. 

    """
    stream = open(conf_file, "r")
    try:
        docs = yaml.load_all(stream, yaml.FullLoader)
    except yaml.scanner.ScannerError as e:
        logger.exception(e)
        exit()
            
    for doc in docs:
        try:
            yield doc
        except TypeError:
            logger.exception("configs.yaml file is empty.")
            exit()
