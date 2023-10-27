"""Validate Config.

A module which defines methods to
verify configuration files.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import os
from wf_psf.utils.read_config import RecursiveNamespace
from wf_psf.utils.validate_config_dicts import validate_dict
import logging

logger = logging.getLogger(__name__)


def check_if_positive_int(*conf_param):
    """Check if Postive Integer.

    A function to check if the configuration
    parameter is a positive integer.

    Parameters
    ----------
    *conf_param: list
        A list containing configuration parameters

    Returns
    -------
    bool
        Returns a boolean
    """

    return bool(isinstance(conf_param[0], int) > 0)


def check_if_positive_float(*conf_param):
    """Check if Postive Float.

    A function to check if the configuration
    parameter is a positive float.

    Parameters
    ----------
    *conf_param: list
        A list containing configuration parameters

    Returns
    -------
    bool
        Returns a boolean
    """
    return bool(isinstance(conf_param[0], float) > 0)


def get_attr_from_RN(rn, key):
    """Get Attribute from RecursiveNamespace.

    A function to return the attribute of a
    RecursiveNamespace object.

    Parameters
    ----------
    rn: RecursiveNamespace instance
        A RecursiveNamespace instance
    key: str
        A string representation of an object in the namespace of the RecursiveNamespace.

    Returns
    -------
    value
        The value of the attribute of an object.

    """
    return rn.__getattribute__(key)


class ValidateConfig:
    """Validate Config Class.

    A class to validate the configuration
    files.

    Parameters
    ----------
    file_handler: object
        An instance of the FileIOHandler
    config_file: RecursiveNamespace object
        RecursiveNamespace object containing the training configuration parameters
    config_type: str
        A string representing the type of configuration file, i.e. data, training, etc.

    """

    def __init__(self, file_handler, config_file, config_type):
        self.file_handler = file_handler
        self.config_file = config_file
        self.validate_dict = validate_dict[config_type]

    def validate_recursiveNamespace(self, rn, keys):
        if not isinstance(keys, dict):
            return self.validate(get_attr_from_RN(rn, keys), keys)
        else:
            for key, v in keys.items():
                if isinstance(get_attr_from_RN(rn, key), RecursiveNamespace):
                    self.validate_recursiveNamespace(
                        get_attr_from_RN(rn, key), keys[key]
                    )
                else:
                    self.validate(get_attr_from_RN(rn, key), v)

        return

    def check_config_exists(self, *conf_param):
        """Check Config Exists.

        A method to check if the config file exists.

        Parameters
        ----------
        conf_param: list
            A list of configuration of parameters

        Returns
        -------
        bool
            Returns a boolean


        """
        return bool(
            os.path.exists(os.path.join(self.file_handler.config_path, conf_param[0]))
        )

    def check_none(self, *conf_param):
        """Check if None.

        A method to check config file is none.

        Parameters
        ----------
        config_param: list
            A list of configuration parameters

        Returns
        -------
        bool
            Returns a boolean
        """
        return bool(conf_param[0] is None)

    def check_if_equals_name(self, *conf_params):
        """Check if Config Param Equals Name.

        A method to check config param equals the
        expected name.

        Parameters
        ----------
        config_params: list
            A list of configuration parameters.

        Returns
        -------
        bool
            Returns a boolean

        """
        return bool(conf_params[0] in conf_params[1])

    def check_if_bool(self, *conf_param):
        """Check If Boolean.

        A method to check config_param type is a
        boolean.

        Parameters
        ----------
        config_param: list
            A list of configuration parameters

        Returns
        -------
        bool
            Returns a boolean

        """
        return bool(type(conf_param[0]) is bool)

    def validate(self, config_param, conditions):
        """Validate.

        A function to validate the config parameter
        according to the condition.
        """

        condition_type = {
            "check_file_exists": self.check_config_exists,
            "check_none": self.check_none,
            "check_if_equals_name": self.check_if_equals_name,
            "check_if_positive_int": check_if_positive_int,
            "check_if_positive_float": check_if_positive_float,
            "check_if_bool": self.check_if_bool,
        }

        logger.info("Evaluating conditions...")
        try:
            for condition in conditions["function"]:
                condition_type[condition]

                if condition_type[condition](config_param, conditions["name"]) is True:
                    logger.info(
                        "{} fulfills condition {}".format(config_param, condition)
                    )
                    logger.info(" ")
                    break
                else:
                    logger.info(
                        "{} does not fulfill condition {}.".format(
                            config_param, condition
                        )
                    )
                    logger.info(" ")
                    continue
        except KeyError:
            logger.exception(
                "Config Param {} does not fulfill conditions.".format(config_param)
            )
