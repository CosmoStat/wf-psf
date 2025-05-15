"""Inference.

A module which provides a set of functions to perform inference
on PSF models. It includes functions to load a trained model, 
perform inference on a dataset of SEDs and positions, and generate a polychromatic PSF.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import os
import glob
import logging
import numpy as np
from wf_psf.psf_models import psf_models, psf_model_loader
import tensorflow as tf


#def prepare_inputs(...): ...
#def generate_psfs(...): ...
#def run_pipeline(...): ...
