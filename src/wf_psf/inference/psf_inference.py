"""Inference.

A module which provides a PSFInference class to perform inference
with trained PSF models. It is able to load a trained model,
perform inference on a dataset of SEDs and positions, and generate polychromatic PSFs.

:Authors: Jennifer Pollack <jennifer.pollack@cea.fr>, Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import os
from pathlib import Path
import numpy as np
from wf_psf.data.data_handler import DataHandler
from wf_psf.utils.read_config import read_conf
from wf_psf.psf_models import psf_models
from wf_psf.psf_models.psf_model_loader import load_trained_psf_model
import tensorflow as tf


class InferenceConfigHandler:
    ids = ("inference_conf",)

    def __init__(self, inference_config_path: str):
        self.inference_config_path = inference_config_path
        self.inference_config = None
        self.training_config = None
        self.data_config = None


    def load_configs(self):
        """Load configuration files based on the inference config."""
        self.inference_config = read_conf(self.inference_config_path)
        self.set_config_paths()
        self.training_config = read_conf(self.trained_model_config_path)

        if self.data_config_path is not None:
            # Load the data configuration
            self.data_conf = read_conf(self.data_config_path)


    def set_config_paths(self):
        """Extract and set the configuration paths."""
        # Set config paths
        config_paths = self.inference_config.inference.configs

        self.trained_model_path = Path(config_paths.trained_model_path)
        self.model_subdir = config_paths.model_subdir
        self.trained_model_config_path = self.trained_model_path / config_paths.trained_model_config_path
        self.data_config_path = config_paths.data_config_path


    @staticmethod
    def overwrite_model_params(training_config=None, inference_config=None):
        """
        Overwrite training model_params with values from inference_config if available.

        Parameters
        ----------
        training_config : RecursiveNamespace
            Configuration object from training phase.
        inference_config : RecursiveNamespace
            Configuration object from inference phase.

        Notes
        -----
        Updates are applied in-place to training_config.training.model_params.
        """
        model_params = training_config.training.model_params
        inf_model_params = inference_config.inference.model_params

        if model_params and inf_model_params:
            for key, value in inf_model_params.__dict__.items():
                if hasattr(model_params, key):
                    setattr(model_params, key, value)

    

class PSFInference:
    """
    Perform PSF inference using a pre-trained WaveDiff model.

    This class handles the setup for PSF inference, including loading configuration
    files, instantiating the PSF simulator and data handler, and preparing the
    input data required for inference.

    Parameters
    ----------
    inference_config_path : str
        Path to the inference configuration YAML file.
    x_field : array-like, optional
        x coordinates in SHE convention.
    y_field : array-like, optional
        y coordinates in SHE convention.
    seds : array-like, optional
        Spectral energy distributions (SEDs).
    """

    def __init__(self, inference_config_path: str, x_field=None, y_field=None, seds=None):

        self.inference_config_path = inference_config_path

        # Inputs for the model
        self.x_field = x_field
        self.y_field = y_field
        self.seds = seds
    
        # Internal caches for lazy-loading
        self._config_handler = None
        self._simPSF = None
        self._data_handler = None
        self._trained_psf_model = None
        self._n_bins_lambda = None
        self._batch_size = None
        self._cycle = None
        self._output_dim = None

        # Initialise PSF Inference engine
        self.engine = None 

    @property
    def config_handler(self):
        if self._config_handler is None:
            self._config_handler = InferenceConfigHandler(self.inference_config_path)
            self._config_handler.load_configs()
        return self._config_handler

    def prepare_configs(self):
        """Prepare the configuration for inference."""
        # Overwrite model parameters with inference config
        self.config_handler.overwrite_model_params(
            self.training_config, self.inference_config
        )

    @property
    def inference_config(self):
        return self.config_handler.inference_config

    @property
    def training_config(self):
        return self.config_handler.training_config

    @property
    def data_config(self):
        return self.config_handler.data_config

    @property
    def simPSF(self):
        if self._simPSF is None:
            self._simPSF = psf_models.simPSF(self.training_config.training.model_params)
        return self._simPSF

    @property
    def data_handler(self):
        if self._data_handler is None:
            # Instantiate the data handler
            self._data_handler = DataHandler(
                dataset_type="inference",
                data_params=self.data_config,
                simPSF=self.simPSF,
                n_bins_lambda=self.n_bins_lambda,
                load_data=False,
                dataset=None,
            )
        return self._data_handler

    @property
    def trained_psf_model(self):
        if self._trained_psf_model is None:
            self._trained_psf_model = self.load_inference_model()
        return self._trained_psf_model

    def load_inference_model(self):
        """Load the trained PSF model based on the inference configuration.""" 
        model_path = self.config_handler.trained_model_path
        model_dir = self.config_handler.model_subdir
        model_name = self.training_config.training.model_params.model_name
        id_name = self.training_config.training.id_name

        weights_path_pattern = os.path.join(
            model_path,
            model_dir,
            f"{model_dir}*_{model_name}*{id_name}_cycle{self.cycle}*"
        )
    
        # Load the trained PSF model
        return load_trained_psf_model(
            self.training_config,
            self.data_config,
            weights_path_pattern,
        )

    @property
    def n_bins_lambda(self):
        if self._n_bins_lambda is None:
            self._n_bins_lambda = self.inference_config.inference.model_params.n_bins_lda
        return self._n_bins_lambda

    @property
    def batch_size(self):
        if self._batch_size is None:
            self._batch_size = self.inference_config.inference.batch_size
            assert self._batch_size > 0, "Batch size must be greater than 0."
        return self._batch_size

    @property
    def cycle(self):
        if self._cycle is None:
            self._cycle = self.inference_config.inference.cycle
        return self._cycle

    @property
    def output_dim(self):
        if self._output_dim is None:
            self._output_dim = self.inference_config.inference.model_params.output_dim
        return self._output_dim

    def _prepare_positions_and_seds(self):
        """Preprocess and return tensors for positions and SEDs."""
        positions = tf.convert_to_tensor(
            np.array([self.x_field, self.y_field]).T, dtype=tf.float32
        )
        self.data_handler.process_sed_data(self.seds)
        sed_data = self.data_handler.sed_data
        return positions, sed_data

    def run_inference(self):
        """Run PSF inference and return the full PSF array."""
        # Prepare the configuration for inference
        self.prepare_configs()

        # Prepare positions and SEDs for inference
        positions, sed_data = self._prepare_positions_and_seds()

        self.engine = PSFInferenceEngine(
            trained_model=self.trained_psf_model,
            batch_size=self.batch_size,
            output_dim=self.output_dim,
        )
        return self.engine.compute_psfs(positions, sed_data)

    def _ensure_psf_inference_completed(self):
        if self.engine is None or self.engine.inferred_psfs is None:
            self.run_inference()

    def get_psfs(self):
        self._ensure_psf_inference_completed()
        return self.engine.get_psfs()

    def get_psf(self, index):
        self._ensure_psf_inference_completed()
        return self.engine.get_psf(index)

class PSFInferenceEngine:
    def __init__(self, trained_model, batch_size: int, output_dim: int):
        self.trained_model = trained_model
        self.batch_size = batch_size
        self.output_dim = output_dim
        self._inferred_psfs = None

    @property
    def inferred_psfs(self) -> np.ndarray:
        """Access the cached inferred PSFs, if available."""
        return self._inferred_psfs

    def compute_psfs(self, positions: tf.Tensor, sed_data: tf.Tensor) -> np.ndarray:
        """Compute and cache PSFs for the input source parameters."""
        n_samples = positions.shape[0]
        self._inferred_psfs = np.zeros((n_samples, self.output_dim, self.output_dim), dtype=np.float32)

        # Initialize counter
        counter = 0
        while counter < n_samples:
            # Calculate the batch end element
            end = min(counter + self.batch_size, n_samples)

            # Define the batch positions
            batch_pos = positions[counter:end_sample, :]
            batch_seds = sed_data[counter:end_sample, :, :]
            batch_inputs = [batch_pos, batch_seds]
            
            # Generate PSFs for the current batch
            batch_psfs = self.trained_model(batch_inputs, training=False)
            self.inferred_psfs[counter:end, :, :] = batch_psfs.numpy()

            # Update the counter
            counter = end
        
        return self._inferred_psfs

    def get_psfs(self) -> np.ndarray:
        """Get all the generated PSFs."""
        if self._inferred_psfs is None:
            raise ValueError("PSFs not yet computed. Call compute_psfs() first.")
        return self._inferred_psfs

    def get_psf(self, index: int) -> np.ndarray:
        """Get the PSF at a specific index."""
        if self._inferred_psfs is None:
            raise ValueError("PSFs not yet computed. Call compute_psfs() first.")
        return self._inferred_psfs[index]


