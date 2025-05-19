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


class InferenceConfigHandler:
    ids = ("inference_conf",)

    def __init__(
        self,
        trained_model_path: str,
        model_subdir: str,
        training_conf_path: str,
        data_conf_path: str,
        inference_conf_path: str,
    ):
        self.trained_model_path = trained_model_path
        self.model_subdir = model_subdir
        self.training_conf_path = training_conf_path
        self.data_conf_path = data_conf_path
        self.inference_conf_path = inference_conf_path

        # Overwrite the model parameters with the inference configuration
        self.model_params = self.overwrite_model_params(
            self.training_conf, self.inference_conf
        )

    def read_configurations(self):
        # Load the training and data configurations
        self.training_conf = read_conf(training_conf_path)
        self.data_conf = read_conf(data_conf_path)
        self.inference_conf = read_conf(inference_conf_path)

    @staticmethod
    def overwrite_model_params(training_conf=None, inference_conf=None):
        """Overwrite model_params of the training_conf with the inference_conf.

        Parameters
        ----------
        training_conf : RecursiveNamespace
            Configuration object containing model parameters and training hyperparameters.
        inference_conf : RecursiveNamespace
            Configuration object containing inference-related parameters.

        """
        model_params = training_conf.training.model_params
        inf_model_params = inference_conf.inference.model_params

        if model_params is not None and inf_model_params is not None:
            for key, value in inf_model_params.__dict__.items():
                # Check if model_params has the attribute
                if hasattr(model_params, key):
                    # Set the attribute of model_params to the new value
                    setattr(model_params, key, value)

        return model_params


class PSFInference:
    """Class to perform inference on PSF models.


    Parameters
    ----------
    inference_conf_path : str
        Path to the inference configuration file.

    """

    def __init__(self, inference_conf_path: str):

        self.inference_conf_path = inference_conf_path
        # Load the training and data configurations
        self.inference_conf = read_conf(inference_conf_path)

        # Set config paths
        self.config_paths = self.inference_conf.inference.configs.config_paths
        self.trained_model_path = self.config_paths.trained_model_path
        self.model_subdir = self.config_paths.model_subdir
        self.training_config_path = self.config_paths.training_config_path
        self.data_conf_path = self.config_paths.data_conf_path

        # Load the training and data configurations
        self.training_conf = read_conf(self.training_conf_path)
        if self.data_conf_path is not None:
            # Load the data configuration
            self.data_conf = read_conf(self.data_conf_path)
        else:
            self.data_conf = None

        # Set source parameters
        self.x_field = None
        self.y_field = None
        self.seds = None
        self.trained_psf_model = None

        # Set compute PSF placeholder
        self.inferred_psfs = None

        # Set the number of labmda bins
        self.n_bins_lambda = self.inference_conf.inference.model_params.n_bins_lda
        # Set the batch size
        self.batch_size = self.inference_conf.inference.batch_size
        assert self.batch_size > 0, "Batch size must be greater than 0."
        # Set the cycle to use for inference
        self.cycle = self.inference_conf.inference.cycle
        # Get output psf dimensions
        self.output_dim = self.inference_conf.inference.model_params.output_dim

        # Overwrite the model parameters with the inference configuration
        self.training_conf.training.model_params = self.overwrite_model_params(
            self.training_conf, self.inference_conf
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
            dataset=None,
        )

        # Load the trained PSF model
        self.trained_psf_model = self.get_trained_psf_model()

    @staticmethod
    def overwrite_model_params(training_conf=None, inference_conf=None):
        """Overwrite model_params of the training_conf with the inference_conf.

        Parameters
        ----------
        training_conf : RecursiveNamespace
            Configuration object containing model parameters and training hyperparameters.
        inference_conf : RecursiveNamespace
            Configuration object containing inference-related parameters.

        """
        model_params = training_conf.training.model_params
        inf_model_params = inference_conf.inference.model_params
        if model_params is not None and inf_model_params is not None:
            for key, value in inf_model_params.__dict__.items():
                # Check if model_params has the attribute
                if hasattr(model_params, key):
                    # Set the attribute of model_params to the new value
                    setattr(model_params, key, value)

        return model_params

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
        self.data_handler.process_sed_data(seds)
        self.sed_data = self.data_handler.sed_data

    def compute_psfs(self):
        """Compute the PSFs for the input source parameters."""

        # Check if source parameters are set
        if self.positions is None or self.sed_data is None:
            raise ValueError(
                "Source parameters not set. Call set_source_parameters first."
            )

        # Get the number of samples
        n_samples = self.positions.shape[0]
        # Initialize counter
        counter = 0
        # Initialize PSF array
        self.inferred_psfs = np.zeros((n_samples, self.output_dim, self.output_dim))

        while counter < n_samples:
            # Calculate the batch end element
            if counter + self.batch_size <= n_samples:
                end_sample = counter + self.batch_size
            else:
                end_sample = n_samples

            # Define the batch positions
            batch_pos = self.positions[counter:end_sample, :]
            batch_seds = self.sed_data[counter:end_sample, :, :]

            # Generate PSFs for the current batch
            batch_inputs = [batch_pos, batch_seds]
            batch_poly_psfs = self.trained_psf_model(batch_inputs, training=False)

            # Append to the PSF array
            self.inferred_psfs[counter:end_sample, :, :] = batch_poly_psfs.numpy()

            # Update the counter
            counter += self.batch_size

    def get_psfs(self) -> np.ndarray:
        """Get all the generated PSFs.

        Returns
        -------
        np.ndarray
            The generated PSFs for the input source parameters.
            Shape is (n_samples, output_dim, output_dim).
        """
        if self.inferred_psfs is None:
            self.compute_psfs()
        return self.inferred_psfs

    def get_psf(self, index) -> np.ndarray:
        """Generate the generated PSF at a specific index.

        Returns
        -------
        np.ndarray
            The generated PSFs for the input source parameters.
            Shape is (output_dim, output_dim).
        """
        if self.inferred_psfs is None:
            self.compute_psfs()
        return self.inferred_psfs[index]
