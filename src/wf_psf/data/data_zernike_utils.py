"""Utilities for Zernike Data Handling.

This module provides utility functions for working with Zernike coefficients, including:
- Prior generation
- Data loading
- Conversions between physical displacements (e.g., defocus, centroid shifts) and modal Zernike coefficients

Useful in contexts where Zernike representations are used to model optical aberrations or link physical misalignments to wavefront modes.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr>

"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import tensorflow as tf
from wf_psf.data.centroids import compute_centroid_correction
from wf_psf.instrument.ccd_misalignments import compute_ccd_misalignment
from wf_psf.utils.read_config import RecursiveNamespace
import logging

logger = logging.getLogger(__name__)


@dataclass
class ZernikeInputs:
    zernike_prior: Optional[np.ndarray]  # true prior, if provided (e.g. from PDC)
    centroid_dataset: Optional[Union[dict, 'RecursiveNamespace']]   # only used in training/simulation
    misalignment_positions: Optional[np.ndarray]  # needed for CCD corrections


class ZernikeInputsFactory:
    @staticmethod
    def build(data, run_type: str, model_params, prior: Optional[np.ndarray] = None) -> ZernikeInputs:
        """Builds a ZernikeInputs dataclass instance based on run type and data.

        Parameters
        ----------
        data : Union[dict, DataConfigHandler]
            Dataset object containing star positions, priors, and optionally pixel data.
        run_type : str
            One of 'training', 'simulation', or 'inference'.
        model_params : RecursiveNamespace
            Model parameters, including flags for prior/corrections.
        prior : Optional[np.ndarray]
            An explicitly passed prior (overrides any inferred one if provided).

        Returns
        -------
        ZernikeInputs
        """
        centroid_dataset = None
        positions = None

        if run_type in {"training", "simulation"}:
            centroid_dataset = data  # Assuming data is a DataConfigHandler or similar object containing train and test datasets
            positions = np.concatenate(
                [
                    data.training_data.dataset["positions"],
                    data.test_data.dataset["positions"]
                ],
                axis=0,
            )

            if model_params.use_prior:
                if prior is not None:
                    logger.warning(
                        "Zernike prior explicitly provided; ignoring dataset-based prior despite use_prior=True."
                    )
                else:
                    prior = get_np_zernike_prior(data)

        elif run_type == "inference":
            centroid_dataset = None
            positions = data["positions"]

            if model_params.use_prior:
                # Try to extract prior from `data`, if present
                prior = getattr(data, "zernike_prior", None) if not isinstance(data, dict) else data.get("zernike_prior")

                if prior is None:
                    logger.warning(
                        "model_params.use_prior=True but no prior found in inference data. Proceeding with None."
                    )

        else:
            raise ValueError(f"Unsupported run_type: {run_type}")

        return ZernikeInputs(
            zernike_prior=prior,
            centroid_dataset=centroid_dataset,
            misalignment_positions=positions
        )


def get_np_zernike_prior(data):
    """Get the zernike prior from the provided dataset.

    This method concatenates the stars from both the training
    and test datasets to obtain the full prior.

    Parameters
    ----------
    data : DataConfigHandler
        Object containing training and test datasets.

    Returns
    -------
    zernike_prior : np.ndarray
        Numpy array containing the full prior.
    """
    zernike_prior = np.concatenate(
        (
            data.training_data.dataset["zernike_prior"],
            data.test_data.dataset["zernike_prior"],
        ),
        axis=0,
    )

    return zernike_prior

def pad_contribution_to_order(contribution: np.ndarray, max_order: int) -> np.ndarray:
    """Pad a Zernike contribution array to the max Zernike order."""
    current_order = contribution.shape[1]
    pad_width = ((0, 0), (0, max_order - current_order))
    return np.pad(contribution, pad_width=pad_width, mode="constant", constant_values=0)

def combine_zernike_contributions(contributions: list[np.ndarray]) -> np.ndarray:
    """Combine multiple Zernike contributions, padding each to the max order before summing."""
    if not contributions:
        raise ValueError("No contributions provided.")

    max_order = max(contrib.shape[1] for contrib in contributions)
    n_samples = contributions[0].shape[0]
    if any(c.shape[0] != n_samples for c in contributions):
        raise ValueError("All contributions must have the same number of samples.")

    combined = np.zeros((n_samples, max_order), dtype=np.float32)
    for contrib in contributions:
        padded = pad_contribution_to_order(contrib, max_order)
        combined += padded

    return combined

def assemble_zernike_contributions(
    model_params,
    zernike_prior=None,
    centroid_dataset=None,
    positions=None,
    batch_size=16,
):
    """
    Assemble the total Zernike contribution map by combining the prior,
    centroid correction, and CCD misalignment correction.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Parameters controlling which contributions to apply.
    zernike_prior : Optional[np.ndarray or tf.Tensor]
        The precomputed Zernike prior (e.g., from PDC or another model).
    centroid_dataset : Optional[object]
        Dataset used to compute centroid correction. Must have both training and test sets.
    positions : Optional[np.ndarray or tf.Tensor]
        Positions used for computing CCD misalignment. Must be available in inference mode.
    batch_size : int
        Batch size for centroid correction.

    Returns
    -------
    tf.Tensor
        A tensor representing the full Zernike contribution map.
    """

    zernike_contribution_list = []

    # Prior
    if model_params.use_prior and zernike_prior is not None:
        logger.info("Adding Zernike prior...")
        if isinstance(zernike_prior, np.ndarray):
            zernike_prior = tf.convert_to_tensor(zernike_prior, dtype=tf.float32)
        zernike_contribution_list.append(zernike_prior)
    else:
        logger.info("Skipping Zernike prior (not used or not provided).")

    # Centroid correction (tip/tilt)
    if model_params.correct_centroids and centroid_dataset is not None:
        logger.info("Computing centroid correction...")
        centroid_correction = compute_centroid_correction(
            model_params, centroid_dataset, batch_size=batch_size
        )
        zernike_contribution_list.append(
            tf.convert_to_tensor(centroid_correction, dtype=tf.float32)
        )
    else:
        logger.info("Skipping centroid correction (not enabled or no dataset).")

    # CCD misalignment (focus term)
    if model_params.add_ccd_misalignments and positions is not None:
        logger.info("Computing CCD misalignment correction...")
        ccd_misalignment = compute_ccd_misalignment(model_params, positions)
        zernike_contribution_list.append(
            tf.convert_to_tensor(ccd_misalignment, dtype=tf.float32)
        )
    else:
        logger.info("Skipping CCD misalignment correction (not enabled or no positions).")

    # If no contributions, return zeros tensor to avoid crashes
    if not zernike_contribution_list:
        logger.warning("No Zernike contributions found. Returning zero tensor.")
        # Infer batch size and zernike order from model_params
        n_samples = 1
        n_zks = getattr(model_params.param_hparams, "n_zernikes", 10)
        return tf.zeros((n_samples, n_zks), dtype=tf.float32)

    combined_zernike_prior = combine_zernike_contributions(zernike_contribution_list)

    return tf.convert_to_tensor(combined_zernike_prior, dtype=tf.float32)
        


def shift_x_y_to_zk1_2_wavediff(dxy, tel_focal_length=24.5, tel_diameter=1.2):
    """Compute Zernike 1(2) for a given shifts in x(y) in WaveDifff conventions.

    All inputs should be in [m].
    A displacement of, for example, 0.5 pixels should be scaled with the corresponding pixel scale,
    e.g. 12[um], to get a displacement in [m], which would be `dxy=0.5*12e-6`.

    The output zernike coefficient is in [um] units as expected by wavediff.

    To apply match the centroid with a `dx` that has a corresponding `zk1`,
    the new PSF should be generated with `-zk1`.

    The same applies to `dy` and `zk2`.

    Parameters
    ----------
    dxy : float
        Centroid shift in [m]. It can be on the x-axis or the y-axis.
    tel_focal_length : float
        Telescope focal length in [m].
    tel_diameter : float
        Telescope aperture diameter in [m].
    """
    reference_pix_sampling = 12e-6
    zernike_norm_factor = 2.0

    # return zernike_norm_factor * (dx/reference_pix_sampling) / (tel_focal_length * tel_diameter / 2)
    return (
        zernike_norm_factor
        * (tel_diameter / 2)
        * np.sin(np.arctan((dxy / reference_pix_sampling) / tel_focal_length))
        * 3.0
    )

def compute_zernike_tip_tilt(
    star_images: np.ndarray,
    star_masks: Optional[np.ndarray] = None,
    pixel_sampling: float = 12e-6,
    reference_shifts: list[float] = [-1/3, -1/3],
    sigma_init: float = 2.5,
    n_iter: int = 20,
) -> np.ndarray:
    """
    Compute Zernike tip-tilt corrections for a batch of PSF images.

    This function estimates the centroid shifts of multiple PSFs and computes 
    the corresponding Zernike tip-tilt corrections to align them with a reference.

    Parameters
    ----------
    star_images : np.ndarray
        A batch of PSF images (3D array of shape `(num_images, height, width)`).
    star_masks : np.ndarray, optional
        A batch of masks (same shape as `star_postage_stamps`). Each mask can have:
        - `0` to ignore the pixel.
        - `1` to fully consider the pixel.
        - Values in `(0,1]` as weights for partial consideration.
        Defaults to None.
    pixel_sampling : float, optional
        The pixel size in meters. Defaults to `12e-6 m` (12 microns).
    reference_shifts : list[float], optional
        The target centroid shifts in pixels, specified as `[dy, dx]`.  
        Defaults to `[-1/3, -1/3]` (nominal Euclid conditions).
    sigma_init : float, optional
        Initial standard deviation for centroid estimation. Default is `2.5`.
    n_iter : int, optional
        Number of iterations for centroid refinement. Default is `20`.

    Returns
    -------
    np.ndarray
        An array of shape `(num_images, 2)`, where:
        - Column 0 contains `Zk1` (tip) values.
        - Column 1 contains `Zk2` (tilt) values.
    
    Notes
    -----
    - This function processes all images at once using vectorized operations.
    - The Zernike coefficients are computed in the WaveDiff convention.
    """
    from wf_psf.data.centroids import CentroidEstimator
    
    # Vectorize the centroid computation
    centroid_estimator = CentroidEstimator(
                            im=star_images,
                            mask=star_masks, 
                            sigma_init=sigma_init,
                            n_iter=n_iter
                            )

    shifts = centroid_estimator.get_intra_pixel_shifts()

    # Ensure reference_shifts is a NumPy array (if it's not already)
    reference_shifts = np.array(reference_shifts)

    # Reshape to ensure it's a column vector (1, 2)
    reference_shifts = reference_shifts[None,:]
  
    # Broadcast reference_shifts to match the shape of shifts
    reference_shifts = np.broadcast_to(reference_shifts, shifts.shape)  
    
    # Compute displacements
    displacements = (reference_shifts - shifts) # 
    
    # Ensure the correct axis order for displacements (x-axis, then y-axis)
    displacements_swapped = displacements[:, [1, 0]] # Adjust axis order if necessary

    # Call shift_x_y_to_zk1_2_wavediff directly on the vector of displacements
    zk1_2_array = shift_x_y_to_zk1_2_wavediff(displacements_swapped.flatten() * pixel_sampling )  # vectorized call
    
    # Reshape the result back to the original shape of displacements
    zk1_2_array = zk1_2_array.reshape(displacements.shape)
  
    return zk1_2_array


def defocus_to_zk4_zemax(dz, tel_focal_length=24.5, tel_diameter=1.2):
    """Compute Zernike 4 value for a given defocus in zemax conventions.

    All inputs should be in [m].

    Parameters
    ----------
    dz : float
        Shift in the z-axis, perpendicular to the focal plane. Units in [m].
    tel_focal_length : float
        Telescope focal length in [m].
    tel_diameter : float
        Telescope aperture diameter in [m].
    """
    # Base calculation
    zk4 = dz / (8.0 * (tel_focal_length / tel_diameter) ** 2)
    # Apply Z4 normalisation
    # This step depends on the normalisation of the Zernike basis used
    zk4 /= np.sqrt(3)
    # Convert to waves with a reference of 800nm
    zk4 /= 800e-9
    # Remove the peak to valley value
    zk4 /= 2.0

    return zk4


def defocus_to_zk4_wavediff(dz, tel_focal_length=24.5, tel_diameter=1.2):
    """Compute Zernike 4 value for a given defocus in WaveDifff conventions.

    All inputs should be in [m].

    The output zernike coefficient is in [um] units as expected by wavediff.

    Parameters
    ----------
    dz : float
        Shift in the z-axis, perpendicular to the focal plane. Units in [m].
    tel_focal_length : float
        Telescope focal length in [m].
    tel_diameter : float
        Telescope aperture diameter in [m].
    """
    # Base calculation
    zk4 = dz / (8.0 * (tel_focal_length / tel_diameter) ** 2)
    # Apply Z4 normalisation
    # This step depends on the normalisation of the Zernike basis used
    zk4 /= np.sqrt(3)

    # Remove the peak to valley value
    zk4 /= 2.0

    # Change units to [um] as Wavediff uses
    zk4 *= 1e6

    return zk4
