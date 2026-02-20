"""TensorFlow dataset converter for PSF datasets.

This module provides the TensorFlowDatasetConverter class, which handles the conversion of PSF datasets (both dataclass-based and dict-based) into TensorFlow tensors suitable for training, evaluation, and inference. It includes methods for processing SEDs using a PSF simulator and converting various dataset formats into a consistent TensorFlow format.

Author: Jennifer Pollack <jennifer.pollack@cea.fr>
"""

import tensorflow as tf
from wf_psf.psf_models.tf_modules.tf_utils import ensure_tensor
from wf_psf.utils.utils import generate_SED_elems_in_tensorflow


class TensorFlowDatasetConverter:
    """
    Converts datasets to TensorFlow tensors and processes SEDs.

    Use this for converting PSF datasets (SHEPSFDataset, dict) to TensorFlow format.

    Parameters
    ----------
    simPSF : PSFSimulator
        SED encoder for processing spectral energy distributions
    n_bins_lambda : int
        Number of wavelength bins for SED discretization
    """

    def __init__(self, simPSF, n_bins_lambda):
        self.simPSF = simPSF
        self.n_bins_lambda = n_bins_lambda

    def process_seds(self, sed_data):
        """
        Process SEDs using simPSF and convert to TensorFlow tensors.

        This is a core operation that must be performed on all SED data before
        use in training or inference. Converts raw SED arrays into wavelength-
        binned TensorFlow tensors.

        Parameters
        ----------
        sed_data : array_like
            Array of SEDs, shape (N, n_wavelengths) or similar

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
        if sed_data is None:
            raise ValueError("SED data must be provided.")

        processed = [
            generate_SED_elems_in_tensorflow(
                sed, self.simPSF, n_bins=self.n_bins_lambda, tf_dtype=tf.float64
            )
            for sed in sed_data
        ]
        sed_tensor = ensure_tensor(processed, dtype=tf.float32)

        return tf.transpose(sed_tensor, perm=[0, 2, 1])

    def convert_psf_dataset(self, dataset, target_field="images"):
        """
        Convert PSF dataset (dataclass) to TensorFlow dict.

        Parameters
        ----------
        dataset : PSFDataset (dataclass)
            Any PSF dataset with attributes: positions, seds, images, masks
            Examples: SHEPSFDataset, RomanPSFDataset, VRAPSFDataset
        target_field : str
            Which field to use as targets ('images' for real data)

        Returns
        -------
        dict
            Dictionary with TensorFlow tensors:
            - 'positions': tf.Tensor (N, 2)
            - 'stars': tf.Tensor (N, H, W) - targets
            - 'masks': tf.Tensor (N, H, W) - if available
            - 'seds': tf.Tensor (N, n_bins_lambda, n_components)
        """
        result = {
            "positions": ensure_tensor(dataset.positions, dtype=tf.float32),
            "stars": ensure_tensor(getattr(dataset, target_field), dtype=tf.float32),
            "seds": self.process_seds(dataset.seds),
        }

        # Add optional fields
        if hasattr(dataset, "masks") and dataset.masks is not None:
            result["masks"] = ensure_tensor(dataset.masks, dtype=tf.float32)

        if hasattr(dataset, "icov") and dataset.icov is not None:
            result["icov"] = ensure_tensor(dataset.icov, dtype=tf.float32)

        return result

    def convert_dict(self, dataset_dict, seds=None):
        """
        Convert all numpy arrays in dataset dict to TensorFlow tensors in-place.
    
        Parameters
        ----------
        dataset_dict : dict
            Dict with dataset fields. Modified in-place.
        seds : array_like, optional
            Pre-provided SEDs (if not in dataset_dict['SEDs'])
    
        Returns
        -------
        dict
            The same dict (modified in-place) for convenience
        """
        # Process SEDs first (special handling)
        sed_data = seds if seds is not None else dataset_dict.get("SEDs")
        if sed_data is not None:
            dataset_dict["SEDs"] = self.process_seds(sed_data)
    
        # Convert all remaining numpy arrays to tensors
        for key, value in dataset_dict.items():
            if key == "SEDs":
                continue  # Already processed above
        
            if isinstance(value, np.ndarray) and not tf.is_tensor(value):
                dataset_dict[key] = ensure_tensor(value, dtype=tf.float32)
    
        return dataset_dict
    
    def convert_inference_data(self, positions, sources=None, masks=None, seds=None):
        """
        Convert inference data to TensorFlow format.

        Specialized method for inference use case where data is provided
        as separate arrays rather than a dict.

        Parameters
        ----------
        positions : array_like
            Focal plane positions, shape (N, 2)
        sources : array_like, optional
            Source images/stamps, shape (N, H, W)
        masks : array_like, optional
            Quality masks, shape (N, H, W)
        seds : array_like, optional
            Spectral energy distributions

        Returns
        -------
        dict
            TensorFlow tensors ready for inference
        """
        result = {
            "positions": ensure_tensor(positions, dtype=tf.float32),
        }

        if sources is not None:
            result["sources"] = ensure_tensor(sources, dtype=tf.float32)

        if masks is not None:
            result["masks"] = ensure_tensor(masks, dtype=tf.float32)

        if seds is not None:
            result["seds"] = self.process_seds(seds)

        return result
