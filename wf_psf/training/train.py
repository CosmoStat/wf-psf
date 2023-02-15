"""Train.

A module which defines the classes and methods
to manage training of the psf model.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import sys
import numpy as np
import time
import tensorflow as tf
import tensorflow_addons as tfa
from wf_psf.read_config import read_conf
import os
import logging
import wf_psf.io as io
from wf_psf.psf_models import psf_models

logger = logging.getLogger(__name__)


def setup_training():
    """Setup Training.

    A function to setup training.


    """
    device_name = get_gpu_info()
    logger.info("Found GPU at: {}".format(device_name))


class TrainingParamsHandler:
    """Training Parameters Handler.

    A class to handle training parameters accessed:

    Parameters
    ----------
    - training_params: Recursive Namespace
        Recursive namespace object containing training input params
    - id_name: str
        ID name
    - run_id_name: str
        Run ID name
    - output_dirs: FileIOHandler
        FileIOHandler instance
    - saving_optim_hist: dict
        Dictionary storing optimiser parameters

    """

    def __init__(
        self, training_params, output_dirs, id_name="-coherent_euclid_200stars"
    ):
        self.training_params = training_params.training
        self.id_name = id_name
        self.run_id_name = self.model_name + self.id_name
        self.checkpoint_dir = output_dirs.get_checkpoint_dir()
        self.saving_optim_hist = {}

    @property
    def model_name(self):
        """PSF Model Name."""
        return self.training_params.model_params.model_name

    @property
    def model_params(self):
        """PSF Model Params."""
        return self.training_params.model_params

    @property
    def training_hparams(self):
        """Training Hyperparameters."""
        return self.training_params.training_hparams

    @property
    def training_data_params(self):
        """Training Data Params."""
        return self.training_params.data.training

    @property
    def test_data_params(self):
        """Test Data Params."""
        return self.training_params.data.test


def get_gpu_info():
    """Get GPU Information.

    A function to return GPU
    device name.

    Returns
    -------
    device_name
        Name of GPU device

    """
    device_name = tf.test.gpu_device_name()
    return device_name


def train(training_params, output_dirs):
    """Train.

    A function to train the psf model.

    Parameters
    ----------
    training_params: Recursive Namespace
        Recursive Namespace object containing training parameters
    output_dirs: str
        Absolute paths to training output directories

    """

    training_handler = TrainingParamsHandler(training_params, output_dirs)

    psf_model = psf_models.get_psf_model(
        training_handler.model_name,
        training_handler.model_params,
        training_handler.training_hparams,
    )

    logger.info(f"PSF Model class: `{training_handler.model_name}` initialized...")

if __name__ == "__main__":
    workdir = os.getenv("HOME")

    # prtty print
    training_params = read_conf(
        os.path.join(workdir, "Projects/wf-psf/wf_psf/config/training_config.yaml")
    )

    training_params = TrainingParams(training_params)
