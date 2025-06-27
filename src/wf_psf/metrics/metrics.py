"""Metrics.

A module which contains the specific functions
for performing various sets of metrics evaluation
of the trained psf model.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np
import tensorflow as tf
import galsim as gs
import wf_psf.utils.utils as utils
from wf_psf.psf_models.psf_models import build_PSF_model
from wf_psf.sims import psf_simulator as psf_simulator
from wf_psf.training.train_utils import compute_noise_std_from_stars
import logging

logger = logging.getLogger(__name__)


def compute_poly_metric(
    tf_semiparam_field,
    gt_tf_semiparam_field,
    simPSF_np,
    tf_pos,
    tf_SEDs,
    n_bins_lda=20,
    n_bins_gt=20,
    batch_size=16,
    dataset_dict=None,
    mask=False,
):
    """Calculate metrics for polychromatic reconstructions.

    The ``tf_semiparam_field`` should be the model to evaluate, and the
    ``gt_tf_semiparam_field`` should be loaded with the ground truth PSF field.

    Relative values returned in [%] (so multiplied by 100).

    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    gt_tf_semiparam_field: PSF field object
        Ground truth model to produce gt observations at any position
        and wavelength.
    simPSF_np: PSF simulator object
        Simulation object to be used by ``generate_packed_elems`` function.
    tf_pos: Tensor or numpy.ndarray [batch x 2] floats
        Positions to evaluate the model.
    tf_SEDs: numpy.ndarray [batch x SED_samples x 2]
        SED samples for the corresponding positions.
    n_bins_lda: int
        Number of wavelength bins to use for the polychromatic PSF.
    n_bins_gt: int
        Number of wavelength bins to use for the ground truth polychromatic PSF.
    batch_size: int
        Batch size for the PSF calcualtions.
    dataset_dict: dict
        Dictionary containing the dataset information. If provided, and if the `'stars'` key
        is present, the noiseless stars from the dataset are used to compute the metrics.
        Otherwise, the stars are generated from the gt model.
        Default is `None`.
    mask: bool
        If `True`, predictions are masked using the same mask as the target data, ensuring
        that metric calculations consider only unmasked regions.
        Default is `False`.

    Returns
    -------
    rmse: float
        RMSE value.
    rel_rmse: float
        Relative RMSE value. Values in %.
    std_rmse: float
        Standard deviation of RMSEs.
    std_rel_rmse: float
        Standard deviation of relative RMSEs. Values in %.

    """
    # Generate SED data list for the model
    packed_SED_data = [
        utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
        for _sed in tf_SEDs
    ]
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    pred_inputs = [tf_pos, tf_packed_SED_data]

    # Model prediction
    preds = tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    # Ground truth data preparation
    if dataset_dict is None or "stars" not in dataset_dict:
        logger.info(
            "No precomputed ground truth stars found. Regenerating from the ground truth model using configured interpolation settings."
        )
        # Change interpolation parameters for the ground truth simPSF
        simPSF_np.SED_interp_pts_per_bin = 0
        simPSF_np.SED_sigma = 0
        # Generate SED data list for gt model
        packed_SED_data = [
            utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_gt)
            for _sed in tf_SEDs
        ]
        tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
        tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
        pred_inputs = [tf_pos, tf_packed_SED_data]

        # Ground Truth model prediction
        gt_preds = gt_tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    else:
        logger.info("Using precomputed ground truth stars from dataset_dict['stars'].")
        gt_preds = dataset_dict["stars"]

    # If the data is masked, mask the predictions
    if mask:
        logger.info(
            "Applying masks to predictions. Only unmasked regions will be considered for metric calculations."
        )

        masks = 1 - dataset_dict["masks"]

        # Ensure masks as float dtype
        masks = masks.astype(preds.dtype)

        # Weight the mse by the number of unmasked pixels
        weights = np.sum(masks, axis=(1, 2))

        # Avoid divide by zero
        weights = np.maximum(weights, 1e-7)

        # Mask the predictions and ground truth/observations
        preds = preds * masks
        gt_preds = gt_preds * masks
    else:
        weights = np.ones(gt_preds.shape[0]) * gt_preds.shape[1] * gt_preds.shape[2]

    # Calculate residuals
    residuals = np.sqrt(np.sum((gt_preds - preds) ** 2, axis=(1, 2)) / weights)
    gt_star_mean = np.sqrt(np.sum((gt_preds) ** 2, axis=(1, 2)) / weights)

    # RMSE calculations
    rmse = np.mean(residuals)
    rel_rmse = 100.0 * np.mean(residuals / gt_star_mean)

    # STD calculations
    std_rmse = np.std(residuals)
    std_rel_rmse = 100.0 * np.std(residuals / gt_star_mean)

    # Print RMSE values
    logger.info("Absolute RMSE:\t %.4e \t +/- %.4e" % (rmse, std_rmse))
    logger.info("Relative RMSE:\t %.4e %% \t +/- %.4e %%" % (rel_rmse, std_rel_rmse))

    return rmse, rel_rmse, std_rmse, std_rel_rmse


def compute_chi2_metric(
    tf_trained_psf_model,
    gt_tf_psf_model,
    simPSF_np,
    tf_pos,
    tf_SEDs,
    n_bins_lda=20,
    n_bins_gt=20,
    batch_size=16,
    dataset_dict=None,
    mask=False,
):
    """Calculate the chi2 metric for polychromatic reconstructions at observation resolution.

    The ``tf_trained_psf_model`` should be the model to evaluate, and the
    ``gt_tf_psf_model`` should be loaded with the ground truth PSF field.

    Parameters
    ----------
    tf_trained_psf_model: PSF field object
        Trained model to evaluate.
    gt_tf_psf_model: PSF field object
        Ground truth model to produce gt observations at any position
        and wavelength.
    simPSF_np: PSF simulator object
        Simulation object to be used by ``generate_packed_elems`` function.
    tf_pos: Tensor or numpy.ndarray [batch x 2] floats
        Positions to evaluate the model.
    tf_SEDs: numpy.ndarray [batch x SED_samples x 2]
        SED samples for the corresponding positions.
    n_bins_lda: int
        Number of wavelength bins to use for the polychromatic PSF.
    n_bins_gt: int
        Number of wavelength bins to use for the ground truth polychromatic PSF.
    batch_size: int
        Batch size for the PSF calcualtions.
    dataset_dict: dict
        Dictionary containing the dataset information. If provided, and if the `'stars'` key
        is present, the noiseless stars from the dataset are used to compute the metrics.
        Otherwise, the stars are generated from the gt model.
        Default is `None`.
    mask: bool
        If `True`, predictions are masked using the same mask as the target data, ensuring
        that metric calculations consider only unmasked regions.
        Default is `False`.

    Returns
    -------
    reduced_chi2_stat: float
        Reduced chi squared value.
    avg_noise_std_dev: float
        Average estimated noise standard deviation used for the chi squared calculation.

    """
    # Create flag
    noiseless_stars = False

    # Generate SED data list for the model
    packed_SED_data = [
        utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
        for _sed in tf_SEDs
    ]
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    pred_inputs = [tf_pos, tf_packed_SED_data]

    # Model prediction
    preds = tf_trained_psf_model.predict(x=pred_inputs, batch_size=batch_size)

    # Ground truth data preparation
    if dataset_dict is None or "stars" not in dataset_dict:
        logger.info(
            "No precomputed ground truth stars found. Regenerating from the ground truth model using configured interpolation settings."
        )
        # The stars will be noiseless as we are recreating them from the ground truth model
        noiseless_stars = True

        # Change interpolation parameters for the ground truth simPSF
        simPSF_np.SED_interp_pts_per_bin = 0
        simPSF_np.SED_sigma = 0
        # Generate SED data list for gt model
        packed_SED_data = [
            utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_gt)
            for _sed in tf_SEDs
        ]
        tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
        tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
        pred_inputs = [tf_pos, tf_packed_SED_data]

        # Ground Truth model prediction
        reference_stars = gt_tf_psf_model.predict(x=pred_inputs, batch_size=batch_size)

    else:
        logger.info("Using precomputed ground truth stars from dataset_dict['stars'].")
        reference_stars = dataset_dict["stars"]

    # If the data is masked, mask the predictions
    if mask:
        logger.info(
            "Applying masks to predictions. Only unmasked regions will be considered for metric calculations."
        )
        # Change convention
        masks = 1 - dataset_dict["masks"]
        # Ensure masks as float dtype
        masks = masks.astype(preds.dtype)

    else:
        # We create a dummy mask of ones
        masks = np.ones_like(reference_stars, dtype=preds.dtype)

    # Compute noise standard deviation from the reference stars
    if not noiseless_stars:
        estimated_noise_std_dev = compute_noise_std_from_stars(
            reference_stars, masks.astype(bool)
        )
        # Check if there is a zero value
        if np.any(estimated_noise_std_dev == 0):
            logger.info(
                "Chi2 metric calculation: Some estimated standard deviations are zero. Setting them to 1 to avoid division by zero."
            )
            estimated_noise_std_dev[estimated_noise_std_dev == 0] = 1.0
    else:
        # If the stars are noiseless, we set the std dev to 1
        estimated_noise_std_dev = np.ones(reference_stars.shape[0], dtype=preds.dtype)
        logger.info(
            "Using noiseless stars for chi2 calculation. Setting all std dev to 1."
        )

    # Compute residuals
    residuals = (reference_stars - preds) * masks

    # Standardize residuals
    standardized_residuals = np.array(
        [
            (residual - np.sum(residual) / np.sum(mask)) / std_est
            for residual, mask, std_est in zip(
                residuals, masks, estimated_noise_std_dev
            )
        ]
    )
    # Compute the degrees of freedom and the mean
    degrees_of_freedom = np.sum(masks)
    mean_standardized_residuals = np.sum(standardized_residuals) / degrees_of_freedom
    # The degrees of freedom is reduced by 1 because we're removing the mean (see Cochran's theorem)
    reduced_chi2_stat = np.sum(
        ((standardized_residuals - mean_standardized_residuals) * masks) ** 2
    ) / (degrees_of_freedom - 1)

    # Average std deviation
    mean_noise_std_dev = np.mean(estimated_noise_std_dev)

    # Print chi2 values
    logger.info("Reduced chi2:\t %.5e" % (reduced_chi2_stat))
    logger.info("Average noise std dev:\t %.5e" % (mean_noise_std_dev))

    return reduced_chi2_stat, mean_noise_std_dev


def compute_mono_metric(
    tf_semiparam_field,
    gt_tf_semiparam_field,
    simPSF_np,
    tf_pos,
    lambda_list,
    batch_size=32,
):
    """Calculate metrics for monochromatic reconstructions.

    The ``tf_semiparam_field`` should be the model to evaluate, and the
    ``gt_tf_semiparam_field`` should be loaded with the ground truth PSF field.

    Relative values returned in [%] (so multiplied by 100).

    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    gt_tf_semiparam_field: PSF field object
        Ground truth model to produce gt observations at any position
        and wavelength.
    simPSF_np: PSF simulator object
        Simulation object capable of calculating ``phase_N`` values from
        wavelength values.
    tf_pos: list of floats [batch x 2]
        Positions to evaluate the model.
    lambda_list: list of floats [wavelength_values]
        List of wavelength values in [um] to evaluate the model.
    batch_size: int
        Batch size to process the monochromatic PSF calculations.

    Returns
    -------
    rmse_lda: list of float
        List of RMSE as a function of wavelength.
    rel_rmse_lda: list of float
        List of relative RMSE as a function of wavelength. Values in %.
    std_rmse_lda: list of float
        List of standard deviation of RMSEs as a function of wavelength.
    std_rel_rmse_lda: list of float
        List of standard deviation of relative RMSEs as a function of
        wavelength. Values in %.

    """
    # Initialise lists
    rmse_lda = []
    rel_rmse_lda = []
    std_rmse_lda = []
    std_rel_rmse_lda = []

    total_samples = tf_pos.shape[0]

    # Main loop for each wavelength
    for it in range(len(lambda_list)):
        # Set the lambda (wavelength) and the required wavefront N
        lambda_obs = lambda_list[it]
        phase_N = simPSF_np.feasible_N(lambda_obs)

        residuals = np.zeros(total_samples)
        gt_star_mean = np.zeros(total_samples)

        # Total number of epochs
        n_epochs = int(np.ceil(total_samples / batch_size))
        ep_low_lim = 0
        for ep in range(n_epochs):
            # Define the upper limit
            if ep_low_lim + batch_size >= total_samples:
                ep_up_lim = total_samples
            else:
                ep_up_lim = ep_low_lim + batch_size
            # Extract the batch
            batch_pos = tf_pos[ep_low_lim:ep_up_lim, :]

            # Estimate the monochromatic PSFs
            gt_mono_psf = gt_tf_semiparam_field.predict_mono_psfs(
                input_positions=batch_pos, lambda_obs=lambda_obs, phase_N=phase_N
            )

            model_mono_psf = tf_semiparam_field.predict_mono_psfs(
                input_positions=batch_pos, lambda_obs=lambda_obs, phase_N=phase_N
            )

            num_pixels = gt_mono_psf.shape[1] * gt_mono_psf.shape[2]

            residuals[ep_low_lim:ep_up_lim] = (
                np.sum((gt_mono_psf - model_mono_psf) ** 2, axis=(1, 2)) / num_pixels
            )
            gt_star_mean[ep_low_lim:ep_up_lim] = (
                np.sum((gt_mono_psf) ** 2, axis=(1, 2)) / num_pixels
            )

            # Increase lower limit
            ep_low_lim += batch_size

        # Calculate residuals
        residuals = np.sqrt(residuals)
        gt_star_mean = np.sqrt(gt_star_mean)

        # RMSE calculations
        rmse_lda.append(np.mean(residuals))
        rel_rmse_lda.append(100.0 * np.mean(residuals / gt_star_mean))

        # STD calculations
        std_rmse_lda.append(np.std(residuals))
        std_rel_rmse_lda.append(100.0 * np.std(residuals / gt_star_mean))

    return rmse_lda, rel_rmse_lda, std_rmse_lda, std_rel_rmse_lda


def compute_opd_metrics(tf_semiparam_field, gt_tf_semiparam_field, pos, batch_size=16):
    """Compute the OPD metrics.

    Need to handle a batch size to avoid Out-Of-Memory errors with
    the GPUs. This is specially due to the fact that the OPD maps
    have a higher dimensionality than the observed PSFs.

    The OPD RMSE is computed after having removed the mean from the
    different reconstructions. It is computed only on the
    non-obscured elements from the OPD.

    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    gt_tf_semiparam_field: PSF field object
        Ground truth model to produce gt observations at any position
        and wavelength.
    pos: numpy.ndarray [batch x 2]
        Positions at where to predict the OPD maps.
    batch_size: int
        Batch size to process the OPD calculations.

    Returns
    -------
    rmse: float
        Absolute RMSE value.
    rel_rmse: float
        Relative RMSE value.
    rmse_std: float
        Absolute RMSE standard deviation.
    rel_rmse_std: float
        Relative RMSE standard deviation.

    """
    # Get OPD obscurations
    np_obscurations = np.real(gt_tf_semiparam_field.obscurations.numpy())
    # Define total number of samples
    n_samples = pos.shape[0]

    # Initialise batch variables
    opd_batch = None
    gt_opd_batch = None
    counter = 0
    # Initialise result lists
    rmse_vals = np.zeros(n_samples)
    rel_rmse_vals = np.zeros(n_samples)

    while counter < n_samples:
        # Calculate the batch end element
        if counter + batch_size <= n_samples:
            end_sample = counter + batch_size
        else:
            end_sample = n_samples

        # Define the batch positions
        batch_pos = pos[counter:end_sample, :]
        # We calculate a batch of OPDs
        opd_batch = tf_semiparam_field.predict_opd(batch_pos).numpy()
        gt_opd_batch = gt_tf_semiparam_field.predict_opd(batch_pos).numpy()
        # Remove the mean of the OPD
        opd_batch -= np.mean(opd_batch, axis=(1, 2)).reshape(-1, 1, 1)
        gt_opd_batch -= np.mean(gt_opd_batch, axis=(1, 2)).reshape(-1, 1, 1)
        # Obscure the OPDs
        opd_batch *= np_obscurations
        gt_opd_batch *= np_obscurations
        # Generate obscuration mask
        obsc_mask = np_obscurations > 0
        nb_mask_elems = np.sum(obsc_mask)
        # Compute the OPD RMSE with the masked obscurations
        res_opd = np.sqrt(
            np.array(
                [
                    np.sum((im1[obsc_mask] - im2[obsc_mask]) ** 2) / nb_mask_elems
                    for im1, im2 in zip(opd_batch, gt_opd_batch)
                ]
            )
        )
        gt_opd_mean = np.sqrt(
            np.array(
                [np.sum(im2[obsc_mask] ** 2) / nb_mask_elems for im2 in gt_opd_batch]
            )
        )
        # RMSE calculations
        rmse_vals[counter:end_sample] = res_opd
        rel_rmse_vals[counter:end_sample] = 100.0 * (res_opd / gt_opd_mean)

        # Add the results to the lists
        counter += batch_size

    # Calculate final values
    rmse = np.mean(rmse_vals)
    rel_rmse = np.mean(rel_rmse_vals)
    rmse_std = np.std(rmse_vals)
    rel_rmse_std = np.std(rel_rmse_vals)

    # Print RMSE values
    logger.info("Absolute RMSE:\t %.4e \t +/- %.4e" % (rmse, rmse_std))
    logger.info("Relative RMSE:\t %.4e %% \t +/- %.4e %%" % (rel_rmse, rel_rmse_std))

    return rmse, rel_rmse, rmse_std, rel_rmse_std


def compute_shape_metrics(
    tf_semiparam_field,
    gt_tf_semiparam_field,
    simPSF_np,
    SEDs,
    tf_pos,
    n_bins_lda,
    n_bins_gt,
    output_Q=1,
    output_dim=64,
    batch_size=16,
    opt_stars_rel_pix_rmse=False,
    dataset_dict=None,
):
    """Compute the pixel, shape and size RMSE of a PSF model.

    This is done at a specific sampling and output image dimension.
    It is done for polychromatic PSFs so SEDs are needed.

    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    gt_tf_semiparam_field: PSF field object
        Ground truth model to produce gt observations at any position
        and wavelength.
    simPSF_np:
    SEDs: numpy.ndarray [batch x SED_samples x 2]
        SED samples for the corresponding positions.
    tf_pos: Tensor [batch x 2]
        Positions at where to predict the PSFs.
    n_bins_lda: int
        Number of wavelength bins to use for the polychromatic PSF.
    n_bins_gt: int
        Number of wavelength bins to use for the ground truth polychromatic PSF.
    output_Q: int
        Downsampling rate to match the specified telescope's sampling. The value
        of `output_Q` should be equal to `oversampling_rate` in order to have
        the right pixel sampling corresponding to the telescope characteristics
        `pix_sampling`, `tel_diameter`, `tel_focal_length`. The final
        oversampling obtained is `oversampling_rate/output_Q`.
        Default is `1`, so the output psf will be super-resolved by a factor of
        `oversampling_rate`. TLDR: better use `1` and measure shapes on the
        super-resolved PSFs.
    output_dim: int
        Output dimension of the square PSF stamps.
    batch_size: int
        Batch size to process the PSF estimations.
    opt_stars_rel_pix_rmse: bool
        If `True`, the relative pixel RMSE of each star is added to ther saving dictionary.
        The summary statistics are always computed.
        Default is `False`.
    dataset_dict: dict
        Dictionary containing the dataset information. If provided, and if the `'super_res_stars'`
        key is present, the noiseless super resolved stars from the dataset are used to compute
        the metrics. Otherwise, the stars are generated from the gt model.
        Default is `None`.

    Returns
    -------
    result_dict: dict
        Dictionary with all the results.

    """
    # Save original output_Q and output_dim
    original_out_Q = tf_semiparam_field.output_Q
    original_out_dim = tf_semiparam_field.output_dim
    gt_original_out_Q = gt_tf_semiparam_field.output_Q
    gt_original_out_dim = gt_tf_semiparam_field.output_dim

    # Set the required output_Q and output_dim parameters in the models
    tf_semiparam_field.set_output_Q(output_Q=output_Q, output_dim=output_dim)
    gt_tf_semiparam_field.set_output_Q(output_Q=output_Q, output_dim=output_dim)

    # Need to compile the models again
    tf_semiparam_field = build_PSF_model(tf_semiparam_field)
    gt_tf_semiparam_field = build_PSF_model(gt_tf_semiparam_field)

    # Generate SED data list
    packed_SED_data = [
        utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda) for _sed in SEDs
    ]

    # Prepare inputs
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    pred_inputs = [tf_pos, tf_packed_SED_data]

    # PSF model
    predictions = tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    # Ground truth data preparation
    if dataset_dict is None or (
        "super_res_stars" not in dataset_dict and "SR_stars" not in dataset_dict
    ):
        logger.info(
            "No pre-computed super-resolved ground truth stars found.  Regenerating ground truth super resolved stars from the ground-truth model using configured interpolation settings."
        )
        # Change interpolation parameters for the ground truth simPSF
        simPSF_np.SED_interp_pts_per_bin = 0
        simPSF_np.SED_sigma = 0
        # Generate SED data list for gt model
        packed_SED_data = [
            utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_gt)
            for _sed in SEDs
        ]

        # Prepare inputs
        tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
        tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
        pred_inputs = [tf_pos, tf_packed_SED_data]

        # Ground Truth model
        gt_predictions = gt_tf_semiparam_field.predict(
            x=pred_inputs, batch_size=batch_size
        )

    else:
        logger.info("Using precomputed super-resolved ground truth stars from dataset.")
        if "super_res_stars" in dataset_dict:
            gt_predictions = dataset_dict["super_res_stars"]
        elif "SR_stars" in dataset_dict:
            gt_predictions = dataset_dict["SR_stars"]

    # Calculate residuals
    residuals = np.sqrt(np.mean((gt_predictions - predictions) ** 2, axis=(1, 2)))
    gt_star_mean = np.sqrt(np.mean((gt_predictions) ** 2, axis=(1, 2)))

    # Pixel RMSE for each star
    if opt_stars_rel_pix_rmse:
        stars_rel_pix_rmse = 100.0 * residuals / gt_star_mean

    # RMSE calculations
    pix_rmse = np.mean(residuals)
    rel_pix_rmse = 100.0 * np.mean(residuals / gt_star_mean)

    # STD calculations
    pix_rmse_std = np.std(residuals)
    rel_pix_rmse_std = 100.0 * np.std(residuals / gt_star_mean)

    # Print pixel RMSE values
    logger.info(
        "\nPixel star absolute RMSE:\t %.4e \t +/- %.4e " % (pix_rmse, pix_rmse_std)
    )
    logger.info(
        "Pixel star relative RMSE:\t %.4e %% \t +/- %.4e %%"
        % (rel_pix_rmse, rel_pix_rmse_std)
    )

    # Measure shapes of the reconstructions
    pred_moments = [
        gs.hsm.FindAdaptiveMom(gs.Image(_pred), strict=False) for _pred in predictions
    ]

    # Measure shapes of the reconstructions
    gt_pred_moments = [
        gs.hsm.FindAdaptiveMom(gs.Image(_pred), strict=False)
        for _pred in gt_predictions
    ]

    pred_e1_HSM, pred_e2_HSM, pred_R2_HSM = [], [], []
    gt_pred_e1_HSM, gt_pred_e2_HSM, gt_pred_R2_HSM = [], [], []

    for it in range(len(gt_pred_moments)):
        if (
            pred_moments[it].moments_status == 0
            and gt_pred_moments[it].moments_status == 0
        ):
            pred_e1_HSM.append(pred_moments[it].observed_shape.g1)
            pred_e2_HSM.append(pred_moments[it].observed_shape.g2)
            pred_R2_HSM.append(2 * (pred_moments[it].moments_sigma ** 2))

            gt_pred_e1_HSM.append(gt_pred_moments[it].observed_shape.g1)
            gt_pred_e2_HSM.append(gt_pred_moments[it].observed_shape.g2)
            gt_pred_R2_HSM.append(2 * (gt_pred_moments[it].moments_sigma ** 2))

    pred_e1_HSM = np.array(pred_e1_HSM)
    pred_e2_HSM = np.array(pred_e2_HSM)
    pred_R2_HSM = np.array(pred_R2_HSM)

    gt_pred_e1_HSM = np.array(gt_pred_e1_HSM)
    gt_pred_e2_HSM = np.array(gt_pred_e2_HSM)
    gt_pred_R2_HSM = np.array(gt_pred_R2_HSM)

    # Calculate metrics

    # e1
    e1_res = gt_pred_e1_HSM - pred_e1_HSM
    e1_res_rel = (gt_pred_e1_HSM - pred_e1_HSM) / gt_pred_e1_HSM

    rmse_e1 = np.sqrt(np.mean(e1_res**2))
    rel_rmse_e1 = 100.0 * np.sqrt(np.mean(e1_res_rel**2))
    std_rmse_e1 = np.std(e1_res)
    std_rel_rmse_e1 = 100.0 * np.std(e1_res_rel)

    # e2
    e2_res = gt_pred_e2_HSM - pred_e2_HSM
    e2_res_rel = (gt_pred_e2_HSM - pred_e2_HSM) / gt_pred_e2_HSM

    rmse_e2 = np.sqrt(np.mean(e2_res**2))
    rel_rmse_e2 = 100.0 * np.sqrt(np.mean(e2_res_rel**2))
    std_rmse_e2 = np.std(e2_res)
    std_rel_rmse_e2 = 100.0 * np.std(e2_res_rel)

    # R2
    R2_res = gt_pred_R2_HSM - pred_R2_HSM

    rmse_R2_meanR2 = np.sqrt(np.mean(R2_res**2)) / np.mean(gt_pred_R2_HSM)
    std_rmse_R2_meanR2 = np.std(R2_res / gt_pred_R2_HSM)

    # Print shape/size errors
    logger.info("\nsigma(e1) RMSE =\t\t %.4e \t +/- %.4e " % (rmse_e1, std_rmse_e1))
    logger.info("sigma(e2) RMSE =\t\t %.4e \t +/- %.4e " % (rmse_e2, std_rmse_e2))
    logger.info(
        "sigma(R2)/<R2> =\t\t %.4e \t +/- %.4e " % (rmse_R2_meanR2, std_rmse_R2_meanR2)
    )

    # Print relative shape/size errors
    logger.info(
        "\nRelative sigma(e1) RMSE =\t %.4e %% \t +/- %.4e %%"
        % (rel_rmse_e1, std_rel_rmse_e1)
    )
    logger.info(
        "Relative sigma(e2) RMSE =\t %.4e %% \t +/- %.4e %%"
        % (rel_rmse_e2, std_rel_rmse_e2)
    )

    # Print number of stars
    logger.info("\nTotal number of stars: \t\t%d" % (len(gt_pred_moments)))
    logger.info(
        "Problematic number of stars: \t%d"
        % (len(gt_pred_moments) - gt_pred_e1_HSM.shape[0])
    )

    # Re-et the original output_Q and output_dim parameters in the models
    tf_semiparam_field.set_output_Q(
        output_Q=original_out_Q, output_dim=original_out_dim
    )
    gt_tf_semiparam_field.set_output_Q(
        output_Q=gt_original_out_Q, output_dim=gt_original_out_dim
    )

    # Need to compile the models again
    tf_semiparam_field = build_PSF_model(tf_semiparam_field)
    gt_tf_semiparam_field = build_PSF_model(gt_tf_semiparam_field)

    # Moment results
    result_dict = {
        "pred_e1_HSM": pred_e1_HSM,
        "pred_e2_HSM": pred_e2_HSM,
        "pred_R2_HSM": pred_R2_HSM,
        "gt_pred_e1_HSM": gt_pred_e1_HSM,
        "gt_ped_e2_HSM": gt_pred_e2_HSM,
        "gt_pred_R2_HSM": gt_pred_R2_HSM,
        "rmse_e1": rmse_e1,
        "std_rmse_e1": std_rmse_e1,
        "rel_rmse_e1": rel_rmse_e1,
        "std_rel_rmse_e1": std_rel_rmse_e1,
        "rmse_e2": rmse_e2,
        "std_rmse_e2": std_rmse_e2,
        "rel_rmse_e2": rel_rmse_e2,
        "std_rel_rmse_e2": std_rel_rmse_e2,
        "rmse_R2_meanR2": rmse_R2_meanR2,
        "std_rmse_R2_meanR2": std_rmse_R2_meanR2,
        "pix_rmse": pix_rmse,
        "pix_rmse_std": pix_rmse_std,
        "rel_pix_rmse": rel_pix_rmse,
        "rel_pix_rmse_std": rel_pix_rmse_std,
        "output_Q": output_Q,
        "output_dim": output_dim,
        "n_bins_lda": n_bins_lda,
    }

    if opt_stars_rel_pix_rmse:
        result_dict["stars_rel_pix_rmse"] = stars_rel_pix_rmse

    return result_dict
