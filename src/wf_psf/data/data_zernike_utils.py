"""Utilities for Zernike Data Handling.

This module provides utilities for converting between physical optical
misalignments (centroid shifts, defocus, CCD alignment effects) and
Zernike coefficients used in wavefront modelling.

It supports assembling Zernike contributions from multiple sources,
including centroid corrections, CCD misalignment terms, and optional
priors, for use in PSF modelling pipelines.

This module does not handle data loading or model execution; it only
provides deterministic transformations between physical parameters and
Zernike representations.

:Authors: Tobias Liaudat <tobias.liaudat@cea.fr> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import tensorflow as tf
from wf_psf.data.data_utils import DatasetContainer
from wf_psf.data.centroids import compute_centroid_correction
from wf_psf.instrument.ccd_misalignments import compute_ccd_misalignment
import logging

logger = logging.getLogger(__name__)


@dataclass
class ZernikeDataset:
    """
    Domain-specific canonical view over a DatasetContainer PSF modelling with Zernike input layers.

    Provides access to Zernike-related inputs, priors, and datasets for
    centroid and misalignment corrections

    Attributes
    ----------
    container : DatasetContainer
        Container storing the underlying datasets.
    zernike_prior : Optional[np.ndarray]
        True Zernike prior (e.g., from PDC). May be None.
    sources : Optional[np.ndarray]
        2D image stamps of sources for centroid correction. May be None.
    masks : Optional[np.ndarray]
        Masks corresponding to the sources. May be None.
    positions : Optional[np.ndarray]
        Positions for CCD misalignment corrections. May be None.
    """

    container: DatasetContainer

    @property
    def sources(self) -> Optional[np.ndarray]:
        """Return 2D source images (stamps)."""
        return self.container.get("sources")

    @property
    def masks(self) -> Optional[np.ndarray]:
        """Return masks corresponding to source images."""
        return self.container.get("masks")

    @property
    def positions(self) -> Optional[np.ndarray]:
        """Return positions for CCD misalignment correction."""
        return self.container.get("positions")

    @property
    def zernike_prior(self) -> Optional[np.ndarray]:
        """Return Zernike prior."""
        return self.container.get("zernike_prior")

    @property
    def centroid_inputs(self) -> Optional[dict]:
        """Return a dictionary suitable for centroid correction.

        Includes 'stamps' (sources) and optionally 'masks' if available.
        Returns None if sources are not available.
        """
        if self.sources is None:
            return None
        data = {"stamps": self.sources}
        if self.masks is not None:
            data["masks"] = self.masks
        return data


def assemble_zernike_contributions(
    model_params,
    zernike_prior=None,
    centroid_dataset=None,
    positions=None,
    batch_size=16,
):
    """Assemble Zernike contributions from prior, centroid correction, and CCD misalignment.

    This function checks the model parameters to determine which contributions to include, computes each contribution as needed, and combines them into a single Zernike contribution tensor. It handles the logic for when certain contributions are not used or not available, ensuring that the final output is correctly shaped and contains the appropriate information based on the configuration.

    Parameters
    ----------
    model_params : RecursiveNamespace
        Parameters controlling which contributions to apply.
    zernike_prior : Optional[np.ndarray or tf.Tensor]
        The precomputed Zernike prior. Can be either a NumPy array or a TensorFlow tensor.
        If a Tensor, will be converted to NumPy in eager mode.
    centroid_dataset : Optional[object]
        Dataset used to compute centroid correction. Must have both training and validation sets.
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
        if isinstance(zernike_prior, tf.Tensor):
            if tf.executing_eagerly():
                zernike_prior = zernike_prior.numpy()
            else:
                raise RuntimeError(
                    "Zernike prior is a TensorFlow tensor but eager execution is disabled. "
                    "Cannot call `.numpy()` outside of eager mode."
                )

        elif not isinstance(zernike_prior, np.ndarray):
            raise TypeError(
                "Unsupported zernike_prior type. Must be np.ndarray or tf.Tensor."
            )
        zernike_contribution_list.append(zernike_prior)
    else:
        logger.info("Skipping Zernike prior (not used or not provided).")

    # Centroid correction (tip/tilt)
    if model_params.correct_centroids and centroid_dataset is not None:
        logger.info("Computing centroid correction...")
        centroid_correction = compute_centroid_correction(
            model_params, centroid_dataset, batch_size=batch_size
        )
        zernike_contribution_list.append(centroid_correction)
    else:
        logger.info("Skipping centroid correction (not enabled or no dataset).")

    # CCD misalignment (focus term)
    if model_params.add_ccd_misalignments and positions is not None:
        logger.info("Computing CCD misalignment correction...")
        ccd_misalignment = compute_ccd_misalignment(model_params, positions)
        zernike_contribution_list.append(ccd_misalignment)
    else:
        logger.info(
            "Skipping CCD misalignment correction (not enabled or no positions)."
        )

    # If no contributions, return zeros tensor to avoid crashes
    if not zernike_contribution_list:
        logger.warning("No Zernike contributions found. Returning zero tensor.")
        # Infer batch size and zernike order from model_params
        n_samples = 1
        n_zks = getattr(model_params.param_hparams, "n_zernikes", 10)
        return tf.zeros((n_samples, n_zks), dtype=tf.float32)

    combined_zernike_prior = combine_zernike_contributions(zernike_contribution_list)

    return tf.convert_to_tensor(combined_zernike_prior, dtype=tf.float32)


def pad_contribution_to_order(contribution: np.ndarray, max_order: int) -> np.ndarray:
    """Pad a Zernike contribution array to the max Zernike order.

    Parameters
    ----------
    contribution : np.ndarray
        Array of shape (n_samples, n_orders) representing Zernike contributions.
    max_order : int
        Target number of Zernike order; determines the number of columns after padding

    Returns
    -------
    np.ndarray
        Padded array of shape (n_samples, max_order) with zeros appended to the right if `max_order` > current number of orders.
    """
    current_order = contribution.shape[1]
    pad_width = ((0, 0), (0, max_order - current_order))
    return np.pad(contribution, pad_width=pad_width, mode="constant", constant_values=0)


def combine_zernike_contributions(contributions: list[np.ndarray]) -> np.ndarray:
    """Combine multiple Zernike contribution arrays into a single array.

    Each contribution is zero-padded along the second dimension (Zernike order) to
    match the maximum order across inputs, then summed element-wise.

    Parameters
    ----------
    contributions: list[np.ndarray]
        List of arrays of shape (n_samples, n_orders_i), where all arrays must share
        the same number of samples (first dimension) but may differ in Zernike order
        (second dimension).

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, max_order) containing the summed contributions after
        zero-padding

    Raises
    ------
    ValueError
        If the list is empty or contributions have inconsistent number of samples.
    """
    if not contributions:
        raise ValueError("No contributions provided.")

    if len(contributions) == 1:
        return contributions[0]

    max_order = max(contrib.shape[1] for contrib in contributions)
    n_samples = contributions[0].shape[0]

    if any(c.shape[0] != n_samples for c in contributions):
        raise ValueError("All contributions must have the same number of samples.")

    combined = np.zeros((n_samples, max_order))
    # Pad each contribution to the max order and sum them
    for contrib in contributions:
        padded = pad_contribution_to_order(contrib, max_order)
        combined += padded

    return combined


def pad_tf_zernikes(zk_param: tf.Tensor, zk_prior: tf.Tensor, n_zks_total: int):
    """
    Pad the Zernike coefficient tensors to match the specified total number of Zernikes.

    Parameters
    ----------
    zk_param : tf.Tensor
        Zernike coefficients for the parametric part. Shape [batch, n_zks_param, 1, 1].
    zk_prior : tf.Tensor
        Zernike coefficients for the prior part. Shape [batch, n_zks_prior, 1, 1].
    n_zks_total : int
        Total number of Zernikes to pad to.

    Returns
    -------
    padded_zk_param : tf.Tensor
        Padded Zernike coefficients for the parametric part. Shape [batch, n_zks_total, 1, 1].
    padded_zk_prior : tf.Tensor
        Padded Zernike coefficients for the prior part. Shape [batch, n_zks_total, 1, 1].
    """
    pad_num_param = n_zks_total - tf.shape(zk_param)[1]
    pad_num_prior = n_zks_total - tf.shape(zk_prior)[1]

    padded_zk_param = tf.cond(
        tf.not_equal(pad_num_param, 0),
        lambda: tf.pad(zk_param, [(0, 0), (0, pad_num_param), (0, 0), (0, 0)]),
        lambda: zk_param,
    )

    padded_zk_prior = tf.cond(
        tf.not_equal(pad_num_prior, 0),
        lambda: tf.pad(zk_prior, [(0, 0), (0, pad_num_prior), (0, 0), (0, 0)]),
        lambda: zk_prior,
    )

    return padded_zk_param, padded_zk_prior


def shift_x_y_to_zk1_2_wavediff(dxy, tel_focal_length=24.5, tel_diameter=1.2):
    """Compute Zernike 1(2) for a given shifts in x(y) in WaveDiff conventions.

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
    reference_shifts: list[float] = [-1 / 3, -1 / 3],
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
        im=star_images, mask=star_masks, sigma_init=sigma_init, n_iter=n_iter
    )

    shifts = centroid_estimator.get_intra_pixel_shifts()

    # Ensure reference_shifts is a NumPy array (if it's not already)
    reference_shifts = np.array(reference_shifts)

    # Reshape to ensure it's a column vector (1, 2)
    reference_shifts = reference_shifts[None, :]

    # Broadcast reference_shifts to match the shape of shifts
    reference_shifts = np.broadcast_to(reference_shifts, shifts.shape)

    # Compute displacements
    displacements = reference_shifts - shifts

    # Ensure the correct axis order for displacements (x-axis, then y-axis)
    displacements_swapped = displacements[:, [1, 0]]  # Adjust axis order if necessary

    # Call shift_x_y_to_zk1_2_wavediff directly on the vector of displacements
    zk1_2_array = shift_x_y_to_zk1_2_wavediff(
        displacements_swapped.flatten() * pixel_sampling
    )  # vectorized call

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
    """Compute Zernike 4 value for a given defocus in WaveDiff conventions.

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
