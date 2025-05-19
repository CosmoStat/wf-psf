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
from wf_psf.data.data_handler import DataHandler
from wf_psf.psf_models import psf_models
from wf_psf.psf_models.psf_model_loader import load_trained_psf_model
import tensorflow as tf

def prepare_inputs(dataset):

    # Convert dataset to tensorflow Dataset
    dataset["positions"] = tf.convert_to_tensor(dataset["positions"], dtype=tf.float32)
    


def get_trained_psf_model(model_path, model_dir_name, cycle, training_conf, data_conf):

    trained_model_path = model_path
    model_subdir = model_dir_name
    cycle = cycle
   
    model_name = training_conf.training.model_params.model_name
    id_name = training_conf.training.id_name
    
    weights_path_pattern = os.path.join(
    trained_model_path,
    model_subdir,
    (
        f"{model_subdir}*_{model_name}"
        f"*{id_name}_cycle{cycle}*"
        ),
    )
    return load_trained_psf_model(
        training_conf,
        data_conf,
        weights_path_pattern,
    )


def generate_psfs(psf_model, inputs):
    pass
   

def run_pipeline():
    psf_model = get_trained_psf_model(
        model_path,
        model_dir,
        cycle,
        training_conf,
        data_conf
    )
    inputs = prepare_inputs(
        
    )  
    psfs = generate_psfs(
        psf_model,
        inputs,
        batch_size=1,
    )
    return psfs

