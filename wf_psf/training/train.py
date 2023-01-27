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

logger = logging.getLogger(__name__)


class TrainingParams:
    def __init__(self, training_params, id_name='-coherent_euclid_200stars'):
        """
        A class to handle training parameters accessed:

        - training_params: a Recursive Simplenamespace storing 
        training input params
        """

        self.training_params = training_params
        self.id_name = id_name
        self.run_id_name = self.training_params.model + self.id_name
        self.model_save_file = io.get_model_save_file()
        self.saving_optim_hist = {}

    def train(self):
        # Print GPU and tensorflow info
        device_name = tf.test.gpu_device_name()
        logger.info('Found GPU at: {}'.format(device_name))
        logger.info('tf_version: ' + str(tf.__version__))


if __name__ == "__main__":
    workdir = os.getenv('HOME')

    # prtty print
    training_params = read_conf(os.path.join(
        workdir, "Projects/wf-psf/wf_psf/config/training_config.yaml"))

    training_params = TrainingParams(training_params)
