"""
:file: wf_psf/train.py

:date: 18/01/23
:author: jpollack

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
import wf_psf.psf_models.psf_models as psf_models

logger = logging.getLogger(__name__)


class TrainingParamsHandler:
    def __init__(self, training_params, id_name='-coherent_euclid_200stars'):
        """
        A class to handle training parameters accessed:

        - training_params: a Recursive Simplenamespace storing 
        training input params
        """

        self.training_params = training_params.training
        self.id_name = id_name
        self.run_id_name = self.model_name + self.id_name
        self.model_save_file = io.get_model_save_file()
        self.saving_optim_hist = {}

    @property
    def model_name(self):
        """PSF Model Name"""
        return self.training_params.model_params.model_name

    @property
    def model_params(self):
        """PSF Model Params"""
        return self.training_params.model_params

    @property
    def training_hparams(self):
        """Training Hyperparameters"""
        return self.training_params.training_hparams

    @property
    def training_data_params(self):
        """Training Data Params"""
        return self.training_params.data.training

    @property
    def test_data_params(self):
        """Test Data Params"""
        return self.training_params.data.test

    def set_psf_model(self):
        """Instantiate PSF Model Class"""
        psf_class = psf_models.PSF_CLASS[self.model_name]
        return psf_class


def train(training_params):
    # Print GPU and tensorflow info
    device_name = tf.test.gpu_device_name()
    logger.info('Found GPU at: {}'.format(device_name))
    logger.info('tf_version: ' + str(tf.__version__))

    training_handler = TrainingParamsHandler(training_params)

    psf_model = training_handler.set_psf_model()
    


if __name__ == "__main__":
    workdir = os.getenv('HOME')

    # prtty print
    training_params = read_conf(os.path.join(
        workdir, "Projects/wf-psf/wf_psf/config/training_config.yaml"))

    training_params = TrainingParams(training_params)
