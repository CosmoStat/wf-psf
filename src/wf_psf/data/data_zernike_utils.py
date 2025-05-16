"""Utilities for Zernike Data Handling.

This module provides utility functions for working with Zernike coefficients, including:
- Prior generation
- Data loading
- Conversions between physical displacements (e.g., defocus, centroid shifts) and modal Zernike coefficients

Useful in contexts where Zernike representations are used to model optical aberrations or link physical misalignments to wavefront modes.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np
import tensorflow as tf
from typing import Optional
import logging

logger = logging.getLogger(__name__)


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


def get_zernike_prior(model_params, data, batch_size: int=16):
    """Get Zernike priors from the provided dataset.

    This method concatenates the Zernike priors from both the training
    and test datasets.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Object containing parameters for this PSF model class.
    data : DataConfigHandler
        Object containing training and test datasets.
    batch_size : int, optional
        The batch size to use when processing the stars. Default is 16.

    Returns
    -------
    tf.Tensor
        Tensor containing the observed positions of the stars.

    Notes
    -----
    The Zernike prior are obtained by concatenating the Zernike priors
    from both the training and test datasets along the 0th axis.

    """
    # List of zernike contribution
    zernike_contribution_list = []

    if model_params.use_prior:
        logger.info("Reading in Zernike prior into Zernike contribution list...")
        zernike_contribution_list.append(get_np_zernike_prior(data))

    if model_params.correct_centroids:
        logger.info("Adding centroid correction to Zernike contribution list...")
        zernike_contribution_list.append(
            compute_centroid_correction(model_params, data, batch_size)
        )

    if model_params.add_ccd_misalignments:
        logger.info("Adding CCD mis-alignments to Zernike contribution list...")
        zernike_contribution_list.append(compute_ccd_misalignment(model_params, data))

    if len(zernike_contribution_list) == 1:
        zernike_contribution = zernike_contribution_list[0]
    else:
        # Get max zk order
        max_zk_order = np.max(
            np.array(
                [
                    zk_contribution.shape[1]
                    for zk_contribution in zernike_contribution_list
                ]
            )
        )

        zernike_contribution = np.zeros(
            (zernike_contribution_list[0].shape[0], max_zk_order)
        )

        # Pad arrays to get the same length and add the final contribution
        for it in range(len(zernike_contribution_list)):
            current_zk_order = zernike_contribution_list[it].shape[1]
            current_zernike_contribution = np.pad(
                zernike_contribution_list[it],
                pad_width=[(0, 0), (0, int(max_zk_order - current_zk_order))],
                mode="constant",
                constant_values=0,
            )

            zernike_contribution += current_zernike_contribution

    return tf.convert_to_tensor(zernike_contribution, dtype=tf.float32)


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
