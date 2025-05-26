"""Inference.

A module which provides a PSFInference class to perform inference
with trained PSF models. It is able to load a trained model,
perform inference on a dataset of SEDs and positions, and generate polychromatic PSFs.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>, Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import os
import numpy as np
from wf_psf.data.data_handler import DataHandler
from wf_psf.utils.read_config import read_conf
from wf_psf.psf_models import psf_models
from wf_psf.psf_models.psf_model_loader import load_trained_psf_model
import tensorflow as tf


class InferenceConfigHandler:
    ids = ("inference_conf",)

    def __init__(self, inference_conf_path: str):
        self.inference_conf_path = inference_conf_path

        # Load the inference configuration
        self.read_configurations()

        # Overwrite the model parameters with the inference configuration
        self.model_params = self.overwrite_model_params(
            self.training_conf, self.inference_conf
        )

    def read_configurations(self):
        """Read the configuration files."""
        # Load the inference configuration
        self.inference_conf = read_conf(self.inference_conf_path)

        # Set config paths
        self.set_config_paths()
        
        # Load the training and data configurations
        self.training_conf = read_conf(self.training_conf_path)
        
        if self.data_conf_path is not None:
            # Load the data configuration
            self.data_conf = read_conf(self.data_conf_path)
        else:
            self.data_conf = None

    def set_config_paths(self):
        """Extract and set the configuration paths."""
        # Set config paths
        self.config_paths = self.inference_conf.inference.configs.config_paths
        self.trained_model_path = self.config_paths.trained_model_path
        self.model_subdir = self.config_paths.model_subdir
        self.training_config_path = self.config_paths.training_config_path
        self.data_conf_path = self.config_paths.data_conf_path

    def get_configs(self):
        """Get the configurations."""
        return (self.inference_conf, self.training_conf, self.data_conf)

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
    """
    Perform PSF inference using a pre-trained WaveDiff model.

    This class handles the setup for PSF inference, including loading configuration
    files, instantiating the PSF simulator and data handler, and preparing the
    input data required for inference.

    Parameters
    ----------
    inference_conf_path : str, optional
        Path to the inference configuration YAML file. This file should define
        paths and parameters for the inference, training, and data configurations.
    x_field : array-like, optional
        Array of x field-of-view coordinates in the SHE convention to be transformed
        and passed to the WaveDiff model.
    y_field : array-like, optional
        Array of y field-of-view coordinates in the SHE convention to be transformed
        and passed to the WaveDiff model.
    seds : array-like, optional
        Spectral energy distributions (SEDs) for the sources being modeled. These
        will be used as part of the input to the PSF simulator.

    Attributes
    ----------
    inference_config_handler : InferenceConfigHandler
        Handler object to load and parse inference, training, and data configs.
    inference_conf : dict
        Dictionary containing inference configuration settings.
    training_conf : dict
        Dictionary containing training configuration settings.
    data_conf : dict
        Dictionary containing data configuration settings.
    x_field : array-like
        Input x coordinates after transformation (if applicable).
    y_field : array-like
        Input y coordinates after transformation (if applicable).
    seds : array-like
        Input spectral energy distributions.
    trained_psf_model : keras.Model
        Loaded PSF model used for prediction.
    inferred_psfs : array-like or None
        Array of inferred PSF images, populated after inference is performed.
    simPSF : psf_models.simPSF
        PSF simulator instance initialized with training model parameters.
    data_handler : DataHandler
        Data handler configured for inference, used to prepare inputs to the model.
    n_bins_lambda : int
        Number of spectral bins used for PSF simulation (loaded from config).

    Methods
    -------
    load_inference_params()
        Load parameters required for inference, including spectral binning.
    get_trained_psf_model()
        Load and return the trained Keras model for PSF inference.
    run_inference()
        Run the model on the input data and generate predicted PSFs.
    """


    def __init__(self, inference_conf_path: str, x_field=None, y_field=None, seds=None):

        self.inference_config_handler = InferenceConfigHandler(
            inference_conf_path=inference_conf_path
        )

        self.inference_conf, self.training_conf, self.data_conf = (
            self.inference_config_handler.get_configs()
        )

        # Init source parameters
        self.x_field = x_field
        self.y_field = y_field
        self.seds = seds
        self.trained_psf_model = None

        # Init compute PSF placeholder
        self.inferred_psfs = None

        # Load inference parameters
        self.load_inference_params()

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

    def load_inference_params(self):
        """Load the inference parameters from the configuration file."""
        # Set the number of labmda bins
        self.n_bins_lambda = self.inference_conf.inference.model_params.n_bins_lda

        # Set the batch size
        self.batch_size = self.inference_conf.inference.batch_size
        assert self.batch_size > 0, "Batch size must be greater than 0."
        
        # Set the cycle to use for inference
        self.cycle = self.inference_conf.inference.cycle
        
        # Get output psf dimensions
        self.output_dim = self.inference_conf.inference.model_params.output_dim

 
    def get_trained_psf_model(self):
        """Get the trained PSF model."""

        # Load the trained PSF model
        model_path = self.inference_config_handler.trained_model_path
        model_dir_name = self.inference_config_handler
        model_name = self.training_conf.training.model_params.model_name
        id_name = self.training_conf.training.id_name

        weights_path_pattern = os.path.join(
            model_path,
            model_dir_name,
            (f"{model_dir_name}*_{model_name}" f"*{id_name}_cycle{self.cycle}*"),
        )
        return load_trained_psf_model(
            self.training_conf,
            self.data_conf,
            weights_path_pattern,
        )

    def set_source_parameters(self):
        """Set the input source parameters for inferring the PSF.

        Note
        ----
        The input source parameters are expected to be in the WaveDiff format. See the simulated data
        format for more details.

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
            np.array([self.x_field, self.y_field]).T, dtype=tf.float32
        )
        # Process SED data
        self.data_handler.process_sed_data(self.seds)
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
