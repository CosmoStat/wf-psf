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
from read_config import read_conf
import os
import logging

logger = logging.getLogger('train')


class TrainingParams:
    def __init__(self, training_params):
        """
        A class to handle training parameters accessed:

        - training_params: a Recursive Simplenamespace storing 
        training input params
        """

        self.training_params = training_params
        self.id = self.training_params.model + 


if __name__ == "__main__":
    workdir = os.getenv('HOME')

    # prtty print
    training_params = read_conf(os.path.join(
        workdir, "Projects/wf-psf/wf_psf/config/training_config.yaml"))

    training_params = TrainingParams(training_params)
