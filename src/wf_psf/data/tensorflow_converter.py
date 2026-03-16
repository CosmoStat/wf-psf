"""TensorFlow dataset converter for PSF datasets.

This module provides the TensorFlowDatasetConverter class, which handles the conversion of PSF datasets (both dataclass-based and dict-based) into TensorFlow tensors suitable for training, evaluation, and inference. It includes methods for processing SEDs using a PSF simulator and converting various dataset formats into a consistent TensorFlow format.

Author: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import tensorflow as tf
from typing import Optional, Union
from wf_psf.data.constants import (
    DEFAULT_CANONICAL_KEYS,
    OPTIONAL_KEYS as CONST_OPTIONAL_KEYS,
)
from wf_psf.data.data_utils import DatasetContainer
from wf_psf.psf_models.psf_models import PSFSimulator
from wf_psf.psf_models.tf_modules.tf_utils import ensure_tensor
from wf_psf.utils.utils import generate_SED_elems_in_tensorflow


class TensorFlowDatasetConverter:
    """Dataset Converter to TensorFlow tensors."""

    REQUIRED_KEYS = DEFAULT_CANONICAL_KEYS
    OPTIONAL_KEYS = set(CONST_OPTIONAL_KEYS)

    def convert_dataset(
        self,
        dataset: Union[DatasetContainer, dict],
        simPSF: PSFSimulator,
        n_bins_lambda: int,
        required_keys: Optional[tuple[str, ...]] = None,
        optional_keys: Optional[tuple[str, ...]] = None,
    ):
        """Convert dataset container or dict to TensorFlow tensors.

        Required keys must be present. Optional keys are converted if available.

        Parameters
        ----------
        dataset : DatasetContainer or dict
            Source dataset.
        simPSF : PSFSimulator
            PSF simulator used for SED processing.
        n_bins_lambda : int
            Number of wavelength bins for SED conversion.
        required_keys : tuple[str]
            Fields that must be present in the dataset.
        optional_keys : tuple[str]
            Fields to convert if they exist.

        Returns
        -------
        dict
            Dictionary with TensorFlow tensors keyed by canonical field names.

        Raises
        ------
        ValueError
            If any required key is missing from the dataset.
        """
        req_keys = self.REQUIRED_KEYS if required_keys is None else required_keys
        opt_keys = self.OPTIONAL_KEYS if optional_keys is None else optional_keys

        result = dict(dataset)

        # Handle required keys
        for k in req_keys:
            v = dataset.get(k, None)
            if v is None:
                raise ValueError(f"Required dataset field '{k}' is missing.")
            result[k] = (
                self.process_seds(v, simPSF, n_bins_lambda)
                if k == "seds"
                else ensure_tensor(v, dtype=tf.float32)
            )

        # Handle optional keys
        for k in opt_keys:
            v = dataset.get(k, None)
            if v is None:
                continue
            result[k] = ensure_tensor(v, dtype=tf.float32)

        return result

    @staticmethod
    def process_seds(sed_data, simPSF, n_bins_lambda):
        """
        Process SEDs using simPSF and convert to TensorFlow tensors.

        This is a core operation that must be performed on all SED data before
        use in training or inference. Converts raw SED arrays into wavelength-
        binned TensorFlow tensors.

        Parameters
        ----------
        sed_data : array_like
            Array of SEDs, shape (N, n_wavelengths) or similar
        simPSF : PSFSimulator
            PSF simulator used for SED processing.
        n_bins_lambda : int
            Number of wavelength bins for SED processing.

        Returns
        -------
        tf.Tensor
            Processed SED tensor, shape (N, n_bins_lambda, n_components)

        Raises
        ------
        ValueError
            If sed_data is None

        Notes
        -----
        - Uses tf.float64 internally for precision during generation
        - Returns tf.float32 for training efficiency
        - Transposes to shape (N, n_bins_lambda, n_components)
        """
        processed = [
            generate_SED_elems_in_tensorflow(
                sed, simPSF, n_bins=n_bins_lambda, tf_dtype=tf.float64
            )
            for sed in sed_data
        ]
        sed_tensor = ensure_tensor(processed, dtype=tf.float32)

        return tf.transpose(sed_tensor, perm=[0, 2, 1])
