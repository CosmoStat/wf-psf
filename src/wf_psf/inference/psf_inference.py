"""Inference.

A module which provides a PSFInference class to perform inference
with trained PSF models. It is able to load a trained model,
perform inference on a dataset of SEDs and positions, and generate polychromatic PSFs.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import os
import glob
import logging
import numpy as np
from wf_psf.data.data_handler import DataHandler
from wf_psf.utils.read_config import read_conf
from wf_psf.psf_models import psf_models
from wf_psf.psf_models.psf_model_loader import load_trained_psf_model
import tensorflow as tf
from typing import Optional


class PSFInference:
    """Class to perform inference on PSF models."""

    def __init__(
        self,
        trained_model_path: str,
        model_subdir: str,
        cycle: int,
        training_conf_path: str,
        data_conf_path: str,
        batch_size: Optional[int] = None,
    ):
        self.trained_model_path = trained_model_path
        self.model_subdir = model_subdir
        self.cycle = cycle
        self.training_conf_path = training_conf_path
        self.data_conf_path = data_conf_path

        # Set source parameters
        self.x_field = None
        self.y_field = None
        self.seds = None
        self.trained_psf_model = None

        # Load the training and data configurations
        self.training_conf = read_conf(training_conf_path)
        self.data_conf = read_conf(data_conf_path)

        # Set the number of labmda bins
        self.n_bins_lambda = self.training_conf.training.model_params.n_bins_lambda

        # Set the batch size
        self.batch_size = (
            batch_size
            if batch_size is not None
            else self.training_conf.training.model_params.batch_size
        )

        # Instantiate the PSF simulator object
        self.simPSF = psf_models.simPSF(self.training_conf.training.model_params)

        # Instantiate the data handler
        self.data_handler = DataHandler(
            dataset_type="inference",
            data_params=self.data_conf,
            simPSF=self.simPSF,
            n_bins_lambda=self.n_bins_lambda,
            load_data=False,
        )

        # Load the trained PSF model
        self.trained_psf_model = self.get_trained_psf_model()

    def get_trained_psf_model(self):
        """Get the trained PSF model."""

        model_name = self.training_conf.training.model_params.model_name
        id_name = self.training_conf.training.id_name

        weights_path_pattern = os.path.join(
            self.trained_model_path,
            self.model_subdir,
            (f"{self.model_subdir}*_{model_name}" f"*{id_name}_cycle{self.cycle}*"),
        )
        return load_trained_psf_model(
            self.training_conf,
            self.data_conf,
            weights_path_pattern,
        )

    def set_source_parameters(self, x_field, y_field, seds):
        """Set the input source parameters for inferring the PSF.

        Parameters
        ----------
        x_field : array-like
            X coordinates of the sources in WaveDiff format.
        y_field : array-like
            Y coordinates of the sources in WaveDiff format.
        seds : list or array-like
            A list or array of raw SEDs, where each SED is typically a vector of flux values
            or coefficients. These will be processed using the PSF simulator.
            It assumes the standard WaveDiff SED format.

        """
        # Positions array is of shape (n_sources, 2)
        self.positions = tf.convert_to_tensor(
            np.array([x_field, y_field]).T, dtype=tf.float32
        )
        # Process SED data
        self.sed_data = self.data_handler.process_sed_data(seds)

    def get_psfs(self):
        """Generate PSFs on the input source parameters."""

        while counter < n_samples:
            # Calculate the batch end element
            if counter + batch_size <= n_samples:
                end_sample = counter + batch_size
            else:
                end_sample = n_samples

        # Define the batch positions
        batch_pos = pos[counter:end_sample, :]

        inputs = [self.positions, self.sed_data]
        poly_psfs = self.trained_psf_model(inputs, training=False)

        return poly_psfs


# def run_pipeline():
#     psf_model = get_trained_psf_model(
#         model_path, model_dir, cycle, training_conf, data_conf
#     )
#     inputs = prepare_inputs()
#     psfs = generate_psfs(
#         psf_model,
#         inputs,
#         batch_size=1,
#     )
#     return psfs
