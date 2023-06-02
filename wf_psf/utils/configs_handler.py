"""Configs_Handler.

A module which provides general utility methods
to manage the parameters of the config files

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
from wf_psf.utils.read_config import read_conf
from wf_psf.data.training_preprocessing import TrainingDataHandler, TestDataHandler
from wf_psf.training import train
from wf_psf.psf_models import psf_models
from wf_psf.metrics.metrics_interface import evaluate_model
import logging
from wf_psf.utils.io import FileIOHandler
import os


logger = logging.getLogger(__name__)

CONFIG_CLASS = {}


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
    """Set Config Class.

    A function to select the class of
    a configuration from a dictionary.

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
        config_class = CONFIG_CLASS[config_name]
    except KeyError as e:
        logger.exception("Config name entered is invalid. Check your config settings.")
        exit()

    return config_class


def get_run_config(run_config, config_params, file_handler):
    """Get Run Configuration.

    A function to get the configuration
    for a wf-psf run.

    Parameters
    ----------
    run_config: str
        Name of the type of run configuraton
    config_params: str
        Path of the run configuration file
    file_handler: object
        A class instance of FileIOHandler

    Returns
    -------
    config_class: object
        A class instance of the selected Configuration Class.

    """
    config_class = set_run_config(run_config)
    return config_class(config_params, file_handler)


class DataConfigHandler:
    """DataConfigHandler.

    A class to handle data config
    settings.

    Parameters
    ----------
    data_conf: str
        Path of the data configuration file
    training_model_params: Recursive Namespace object
        Recursive Namespace object containing the training model parameters

    """

    def __init__(self, data_conf, training_model_params):
        try:
            self.data_conf = read_conf(data_conf)
        except FileNotFoundError as e:
            logger.exception(e)
            exit()

        self.simPSF = psf_models.simPSF(training_model_params)
        self.training_data = TrainingDataHandler(
            self.data_conf.data.training,
            self.simPSF,
            training_model_params.n_bins_lda,
        )
        self.test_data = TestDataHandler(
            self.data_conf.data.test,
            self.simPSF,
            training_model_params.n_bins_lda,
        )


@register_configclass
class TrainingConfigHandler:
    """TrainingConfigHandler.

    A class to handle training config
    settings.

    Parameters
    ----------
    ids: tuple
        A tuple containing a string id for the Configuration Class
    training_conf: str
        Path of the training configuration file
    file_handler: object
        A instance of the FileIOHandler class

    """

    ids = ("training_conf",)

    def __init__(self, training_conf, file_handler):
        self.training_conf = read_conf(training_conf)
        self.file_handler = file_handler
        try:
            self.data_conf = DataConfigHandler(
                os.path.join(
                    file_handler.config_path, self.training_conf.training.data_config
                ),
                self.training_conf.training.model_params,
            )
        except TypeError as e:
            logger.exception(
                "Invalid type. Check the data_config param in your training.yaml file."
            )
            exit()

        self.checkpoint_dir = file_handler.get_checkpoint_dir(
            self.file_handler._run_output_dir
        )
        self.optimizer_dir = file_handler.get_optimizer_dir(
            self.file_handler._run_output_dir
        )
        if self.training_conf.training.metrics_config is not None:
            try:
                self.metrics_conf = read_conf(
                    os.path.join(
                        file_handler.config_path,
                        self.training_conf.training.metrics_config,
                    )
                )
            except FileNotFoundError as e:
                logger.exception(
                    "Check the metrics_config param in your training_config.yaml file."
                )
                exit()
            except TypeError as e:
                logger.exception(
                    "Invalid type. Check the metrics_config param in your training_config.yaml file."
                )
                exit()
        else:
            logger.info("Performing training only...")

    def run(self):
        """Run.

        A function to run wave-diff according to the
        input configuration.

        """
        psf_model, checkpoint_filepath = train.train(
            self.training_conf.training,
            self.data_conf.training_data,
            self.data_conf.test_data,
            self.checkpoint_dir,
            self.optimizer_dir,
        )

        if self.metrics_conf is not None:
            evaluate_model(
                self.metrics_conf.metrics,
                self.training_conf.training,
                self.data_conf.training_data,
                self.data_conf.test_data,
                psf_model,
                checkpoint_filepath,
                self.file_handler.get_metrics_dir(self.file_handler._run_output_dir),
            )


@register_configclass
class MetricsConfigHandler:
    """MetricsConfigHandler.

    A class to handle metrics config
    settings.

    Parameters
    ----------
    ids: tuple
        A tuple containing a string id for the Configuration Class
    metrics_conf: str
        Path of the metrics configuration file
    file_handler: object
        An instance of the FileIOHandler

    """

    ids = ("metrics_conf",)

    def __init__(self, metrics_conf, file_handler):
        self.metrics_conf = read_conf(metrics_conf)
        self.metrics_dir = file_handler.get_metrics_dir(file_handler._run_output_dir)
        self.training_conf = read_conf(
            os.path.join(
                self.metrics_conf.metrics.trained_model_path,
                self.metrics_conf.metrics.trained_model_config,
            )
        )

        self.checkpoint_filepath = train.filepath_chkp_callback(
            file_handler.get_checkpoint_dir(
                self.metrics_conf.metrics.trained_model_path
            ),
            self.training_conf.training.model_params.model_name,
            self.training_conf.training.id_name,
            self.metrics_conf.metrics.saved_training_cycle,
        )

        try:
            self.data_conf = DataConfigHandler(
                os.path.join(
                    file_handler.config_path, self.training_conf.training.data_config
                ),
                self.training_conf.training.model_params,
            )
        except TypeError as e:
            logger.exception(e)

        self.psf_model = psf_models.get_psf_model(
            self.training_conf.training.model_params,
            self.training_conf.training.training_hparams,
        )

    def run(self):
        """Run.

        A function to run wave-diff according to the
        input configuration.

        """
        evaluate_model(
            self.metrics_conf.metrics,
            self.training_conf.training,
            self.data_conf.training_data,
            self.data_conf.test_data,
            self.psf_model,
            self.checkpoint_filepath,
            self.metrics_dir,
        )


@register_configclass
class PlottingConfigHandler:
    """PlottingConfigHandler.

    A class to handle plotting config
    settings.

    Parameters
    ----------
    ids: tuple
        A tuple containing a string id for the Configuration Class
    """

    ids = ("plotting_conf",)
