"""SimulationDataLoader.

Loads simulation data from .npy files on disk for training/testing.

Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import os
from wf_psf.data.tensorflow_converter import TensorFlowDatasetConverter


class SimulationDataLoader:
    """
    Loads simulation data from .npy files on disk.

    Use this for training/testing with pre-saved simulation datasets.

    Parameters
    ----------
    dataset_type : str
        Indicates the dataset mode ("train", "test", or "inference").
    data_params : RecursiveNamespace
        Configuration object containing dataset parameters (e.g., file paths, preprocessing flags).
    simPSF : PSFSimulator
        An instance of the PSFSimulator class used to encode SEDs into a TensorFlow-compatible format.
    n_bins_lambda : int
        Number of wavelength bins used to discretize SEDs.
    dataset : dict or list, optional
        If provided, uses this pre-loaded dataset instead of triggering automatic loading.
    sed_data : dict or list, optional
        If provided, uses this SED data directly instead of extracting it from the dataset.

    Attributes
    ----------
    dataset_type : str
        Indicates the dataset mode ("train", "test", or "inference").
    data_params : RecursiveNamespace
        Configuration parameters for data access and structure.
    simPSF : PSFSimulator
        Simulator used to transform SEDs into TensorFlow-ready tensors.
    n_bins_lambda : int
        Number of wavelength bins in the SED representation.
    dataset : dict
        Loaded dataset including keys such as 'positions', 'stars', 'noisy_stars', or similar.
    sed_data : tf.Tensor
        TensorFlow-formatted SED data with shape [batch_size, n_bins_lambda, features].

    """

    def __init__(self, dataset_type, data_params, simPSF, n_bins_lambda):
        self.dataset_type = dataset_type
        self.data_params = data_params
        self.simPSF = simPSF
        self.converter = TensorFlowDatasetConverter(simPSF, n_bins_lambda)
        self.n_bins_lambda = n_bins_lambda

        # Require target_field in params
        if not hasattr(data_params, 'target_field'):
            raise ValueError(
                "data_params must specify 'target_field'. "
                "This should be set by DataConfigHandler or in the config file."
            )
        self.target_field = data_params.target_field

        self.dataset = None
        self.sed_data = None

    def load(self):
        """Load .npy file, validate, process SEDs, convert to TensorFlow."""
        self._load_from_disk()
        self._validate_structure()
        self._process_seds()
        self._convert_to_tensorflow()
        return self.dataset, self.sed_data

    def _load_from_disk(self):
        """Load .npy file."""
        self.dataset = np.load(
            os.path.join(self.data_params.data_dir, self.data_params.file),
            allow_pickle=True,
        )[()]

    def _validate_structure(self):
        """Validate dataset structure based on dataset_type."""
        if self.dataset is None:
            raise ValueError("Dataset is None")

        if "positions" not in self.dataset:
            raise ValueError("Dataset missing required field: 'positions'")

        if self.target_field not in self.dataset:
            raise ValueError(
                f"Missing required field '{self.target_field}' in {self.dataset_type} dataset."
            )

    def _convert_to_tensorflow(self):
        """Convert dataset to TensorFlow tensors."""
        self.dataset = self.converter.convert_dict(self.dataset, self.dataset_type)

    def _process_seds(self):
        """Process SEDs from loaded dataset."""
        if "SEDs" not in self.dataset:
            raise ValueError("Dataset missing 'SEDs' field")

        self.sed_data = self.converter.process_seds(self.dataset["SEDs"])
