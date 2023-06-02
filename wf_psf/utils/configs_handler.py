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
        print(config_class)

    print("Config Class dict", CONFIG_CLASS)
    return config_class


def set_run_config(config_name):
    """Set Config Class.

    A function to select a class of
    the configuration from a dictionary.

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

    Inputs
    ------
    run_config
    """
    config_class = set_run_config(run_config)
    return config_class(config_params, file_handler)


class ConfigHandler:
    """ConfigHandler.

    A class to handle different config
    settings.


    """

    def __init__(self, conffile):
        self.conffile = conffile

    pass


class DataConfigHandler:
    """DataConfigHandler.

    A class to handle data config
    settings.

    """

    def __init__(self, data_conf, training_model_params):
        self.simPSF = psf_models.simPSF(training_model_params)
        self.training_data = TrainingDataHandler(
            data_conf.data.training,
            self.simPSF,
            training_model_params.n_bins_lda,
        )
        self.test_data = TestDataHandler(
            data_conf.data.test,
            self.simPSF,
            training_model_params.n_bins_lda,
        )


@register_configclass
class TrainingConfigHandler:
    """TrainingConfigHandler.

    A class to handle training config
    settings.

    """

    ids = ("training_conf",)

    def __init__(self, training_conf, file_handler):
        self.training_conf = read_conf(training_conf)
        self.file_handler = file_handler
        self.data_conf = DataConfigHandler(
            read_conf(os.path.join(self.training_conf.training.data_config_path)),
            self.training_conf.training.model_params,
        )
        self.checkpoint_dir = file_handler.get_checkpoint_dir(
            self.file_handler._run_output_dir
        )
        self.optimizer_dir = file_handler.get_optimizer_dir(
            self.file_handler._run_output_dir
        )
        self.metrics_conf = read_conf(self.training_conf.training.metrics_config_path)
    
    def run(self):
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

        self.data_conf = DataConfigHandler(
            read_conf(os.path.join(self.training_conf.training.data_config_path)),
            self.training_conf.training.model_params,
        )

        self.psf_model = psf_models.get_psf_model(
            self.training_conf.training.model_params,
            self.training_conf.training.training_hparams,
        )

    def run(self):
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

    """

    ids = ("plotting_conf",)
