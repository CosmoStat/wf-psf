"""Training Config Handler.

A module which provides a class to manage the parameters of the training config file.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import os
from wf_psf.utils.configs_handler import ConfigHandler, register_configclass
from wf_psf.data.data_config_handler import DataConfigHandler
from wf_psf.psf_models import psf_models
from wf_psf.utils.read_config import read_conf
from wf_psf.training import train
from wf_psf.metrics.metrics_config_handler import MetricsConfigHandler
import logging


logger = logging.getLogger(__name__)


@register_configclass
class TrainingConfigHandler(ConfigHandler):
    """TrainingConfigHandler.

    A class to handle training configuration
    parameters.

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
        self.data_params = DataConfigHandler(
            os.path.join(
                file_handler.config_path, self.training_conf.training.data_config
            ),
        )
        self.n_bins_lambda = self.training_conf.training.model_params.n_bins_lambda
        self.simPSF = psf_models.simPSF(self.training_conf.training.model_params)
        self.file_handler.copy_conffile_to_output_dir(
            self.training_conf.training.data_config
        )
        self.checkpoint_dir = file_handler.get_checkpoint_dir(
            self.file_handler._run_output_dir
        )
        self.optimizer_dir = file_handler.get_optimizer_dir(
            self.file_handler._run_output_dir
        )
        self.psf_model_dir = file_handler.get_psf_model_dir(
            self.file_handler._run_output_dir
        )

    def run(self):
        """Run.

        A function to run wavediff according to the
        input configuration.

        """
        train.train(
            self.training_conf.training,
            self.data_params,
            self.simPSF,
            self.n_bins_lambda,
            self.checkpoint_dir,
            self.optimizer_dir,
            self.psf_model_dir,
        )

        if self.training_conf.training.metrics_config is not None:
            self.file_handler.copy_conffile_to_output_dir(
                self.training_conf.training.metrics_config
            )

            metrics = MetricsConfigHandler(
                os.path.join(
                    self.file_handler.config_path,
                    self.training_conf.training.metrics_config,
                ),
                self.file_handler,
                self.training_conf,
            )

            metrics.run()
